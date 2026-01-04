"""
Two-Stage LLM Pipeline for ASL Caption Generation

This module implements a two-stage approach to ASL-to-English translation:
- Stage 1: Selection - Choose best gloss from top-k predictions for each position
- Stage 2: Sentence - Construct grammatically correct sentence from selected glosses

The separation allows:
1. Independent measurement of selection accuracy
2. Specialized prompts for each task
3. Better semantic coherence through focused selection
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple


class TwoStagePipeline:
    """
    Orchestrates two-stage LLM processing for ASL caption generation.

    Stage 1: Selection - Given top-k predictions per position, select best gloss
    Stage 2: Sentence - Construct sentence from selected glosses
    """

    def __init__(
        self,
        llm_provider: Any,
        stage1_prompt_path: Optional[Path] = None,
        stage2_prompt_path: Optional[Path] = None,
        stage2_rewrite_prompt_path: Optional[Path] = None,
    ):
        """
        Initialize the two-stage pipeline.

        Args:
            llm_provider: LLM provider instance with generate() method
            stage1_prompt_path: Path to Stage 1 selection prompt template
            stage2_prompt_path: Path to Stage 2 sentence prompt template
            stage2_rewrite_prompt_path: Path to Stage 2 rewrite prompt template
        """
        self.llm = llm_provider

        # Load prompt templates
        prompts_dir = Path(__file__).parent / "prompts"

        self.stage1_template = self._load_prompt(
            stage1_prompt_path or prompts_dir / "llm_prompt_stage1_selection.txt"
        )
        self.stage2_template = self._load_prompt(
            stage2_prompt_path or prompts_dir / "llm_prompt_stage2_sentence.txt"
        )
        self.stage2_rewrite_template = self._load_prompt(
            stage2_rewrite_prompt_path or prompts_dir / "llm_prompt_stage2_sentence_rewrite.txt"
        )

        # Track selections for accuracy measurement
        self.selection_history: List[Dict[str, Any]] = []

    def _load_prompt(self, path: Path) -> str:
        """Load prompt template from file."""
        try:
            return path.read_text(encoding='utf-8')
        except Exception as e:
            print(f"[TwoStagePipeline] Warning: Could not load prompt from {path}: {e}")
            return ""

    # =========================================================================
    # STAGE 1: SELECTION
    # =========================================================================

    def run_stage1_selection(
        self,
        glosses: List[Dict[str, Any]],
        context_sentences: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Stage 1: Select best gloss from top-k for each position.

        Args:
            glosses: List of gloss data with top-k predictions
                Each item: {'gloss': str, 'confidence': float, 'top_k': [...]}
            context_sentences: Optional previous sentences for context

        Returns:
            {
                'selections': ['GLOSS1', 'GLOSS2', ...],
                'reasoning': str,
                'raw_response': str,
                'success': bool
            }
        """
        if not self.stage1_template:
            # Fallback: use top-1 predictions
            return {
                'selections': [g.get('gloss', g.get('top_k', [{}])[0].get('gloss', 'UNKNOWN')) for g in glosses],
                'reasoning': 'Fallback to top-1 (no prompt template)',
                'raw_response': '',
                'success': False
            }

        # Build context section
        context_section = ""
        if context_sentences:
            context_section = "CONTEXT (previous sentences):\n"
            for i, sent in enumerate(context_sentences[-2:], 1):
                context_section += f"  {i}. \"{sent}\"\n"
            context_section += "\nConsider this context when selecting glosses.\n"

        # Format gloss details
        gloss_details = self._format_gloss_details_for_selection(glosses)

        # Build prompt
        prompt = self.stage1_template.replace('{context_section}', context_section)
        prompt = prompt.replace('{gloss_details}', gloss_details)

        try:
            # Call LLM
            response = self.llm.generate(prompt)

            # Parse response
            result = self._parse_selection_response(response, glosses)
            result['raw_response'] = response
            result['success'] = True

            # Record for accuracy tracking
            self._record_selection(glosses, result['selections'])

            return result

        except Exception as e:
            print(f"[TwoStagePipeline] Stage 1 error: {e}")
            # Fallback to top-1
            return {
                'selections': [g.get('gloss', 'UNKNOWN') for g in glosses],
                'reasoning': f'Fallback to top-1 (error: {e})',
                'raw_response': '',
                'success': False
            }

    def _format_gloss_details_for_selection(self, glosses: List[Dict[str, Any]]) -> str:
        """Format glosses for Stage 1 selection prompt."""
        details = []
        for i, g in enumerate(glosses, 1):
            top_k = g.get('top_k', [])
            if top_k:
                detail = f"Position {i}:\n"
                for j, pred in enumerate(top_k[:3], 1):
                    conf = pred.get('confidence', 0) * 100
                    detail += f"  Option {j}: '{pred.get('gloss', 'UNKNOWN')}' ({conf:.1f}%)\n"
                details.append(detail)
            else:
                # Single prediction, no top-k
                conf = g.get('confidence', 0) * 100
                gloss = g.get('gloss', 'UNKNOWN')
                details.append(f"Position {i}:\n  Option 1: '{gloss}' ({conf:.1f}%)\n")
        return "".join(details)

    def _parse_selection_response(
        self,
        response: str,
        glosses: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Parse Stage 1 LLM response to extract selections."""
        response_text = self._clean_json_response(response)

        try:
            result = json.loads(response_text)
            selections = result.get('selections', [])
            reasoning = result.get('reasoning', '')

            # Validate selection count
            if len(selections) != len(glosses):
                print(f"[TwoStagePipeline] Warning: Expected {len(glosses)} selections, got {len(selections)}")
                # Pad or truncate
                while len(selections) < len(glosses):
                    selections.append(glosses[len(selections)].get('gloss', 'UNKNOWN'))
                selections = selections[:len(glosses)]

            return {'selections': selections, 'reasoning': reasoning}

        except json.JSONDecodeError:
            # Try regex extraction
            try:
                match = re.search(r'"selections"\s*:\s*\[(.*?)\]', response_text, re.DOTALL)
                if match:
                    items = re.findall(r'"([^"]+)"', match.group(1))
                    return {'selections': items, 'reasoning': 'Parsed via regex'}
            except Exception:
                pass

            # Fallback to top-1
            return {
                'selections': [g.get('gloss', 'UNKNOWN') for g in glosses],
                'reasoning': 'Fallback to top-1 (parse error)'
            }

    # =========================================================================
    # STAGE 2: SENTENCE CONSTRUCTION
    # =========================================================================

    def run_stage2_sentence(
        self,
        selected_glosses: List[str],
        context_sentences: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Stage 2: Construct sentence from selected glosses.

        Args:
            selected_glosses: List of selected gloss strings from Stage 1
            context_sentences: Optional previous sentences for context

        Returns:
            {
                'sentence': str,
                'used_glosses': List[str],
                'dropped_glosses': List[str],
                'raw_response': str,
                'success': bool
            }
        """
        if not self.stage2_template:
            # Fallback: join glosses
            return {
                'sentence': ' '.join(selected_glosses),
                'used_glosses': selected_glosses,
                'dropped_glosses': [],
                'raw_response': '',
                'success': False
            }

        # Build context section
        context_section = ""
        if context_sentences:
            context_section = "CONTEXT (previous sentences - maintain continuity):\n"
            for i, sent in enumerate(context_sentences[-2:], 1):
                context_section += f"  {i}. \"{sent}\"\n"
            context_section += "\n"

        # Format selected glosses
        glosses_str = json.dumps(selected_glosses)

        # Build prompt
        prompt = self.stage2_template.replace('{context_section}', context_section)
        prompt = prompt.replace('{selected_glosses}', glosses_str)

        try:
            # Call LLM
            response = self.llm.generate(prompt)

            # Parse response
            result = self._parse_sentence_response(response, selected_glosses)
            result['raw_response'] = response
            result['success'] = True

            return result

        except Exception as e:
            print(f"[TwoStagePipeline] Stage 2 error: {e}")
            return {
                'sentence': ' '.join(selected_glosses),
                'used_glosses': selected_glosses,
                'dropped_glosses': [],
                'raw_response': '',
                'success': False
            }

    def run_stage2_sentence_rewrite(
        self,
        selected_glosses: List[str],
        running_caption: str,
        reconstruction_glosses: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Stage 2 with rewrite: Construct sentence AND rewrite full caption.

        Used at reconstruction window boundaries.

        Args:
            selected_glosses: Latest window's selected glosses
            running_caption: Current full caption text
            reconstruction_glosses: All glosses in reconstruction window

        Returns:
            {
                'new_sentence': str,
                'full_caption': str,
                'used_glosses': List[str],
                'dropped_glosses': List[str],
                'raw_response': str,
                'success': bool
            }
        """
        if not self.stage2_rewrite_template:
            # Fallback
            new_sentence = ' '.join(selected_glosses)
            full_caption = f"{running_caption} {new_sentence}".strip()
            return {
                'new_sentence': new_sentence,
                'full_caption': full_caption,
                'used_glosses': selected_glosses,
                'dropped_glosses': [],
                'raw_response': '',
                'success': False
            }

        # Format inputs
        reconstruction_str = json.dumps(reconstruction_glosses or selected_glosses)
        selected_str = json.dumps(selected_glosses)

        # Build prompt
        prompt = self.stage2_rewrite_template.replace('{running_caption}', running_caption or "")
        prompt = prompt.replace('{reconstruction_glosses}', reconstruction_str)
        prompt = prompt.replace('{selected_glosses}', selected_str)

        try:
            # Call LLM
            response = self.llm.generate(prompt)

            # Parse response
            result = self._parse_rewrite_response(response, selected_glosses, running_caption)
            result['raw_response'] = response
            result['success'] = True

            return result

        except Exception as e:
            print(f"[TwoStagePipeline] Stage 2 rewrite error: {e}")
            new_sentence = ' '.join(selected_glosses)
            return {
                'new_sentence': new_sentence,
                'full_caption': f"{running_caption} {new_sentence}".strip(),
                'used_glosses': selected_glosses,
                'dropped_glosses': [],
                'raw_response': '',
                'success': False
            }

    def _parse_sentence_response(
        self,
        response: str,
        selected_glosses: List[str]
    ) -> Dict[str, Any]:
        """Parse Stage 2 sentence response."""
        response_text = self._clean_json_response(response)

        try:
            result = json.loads(response_text)
            return {
                'sentence': result.get('sentence', ' '.join(selected_glosses)),
                'used_glosses': result.get('used_glosses', selected_glosses),
                'dropped_glosses': result.get('dropped_glosses', [])
            }
        except json.JSONDecodeError:
            # Try regex
            try:
                match = re.search(r'"sentence"\s*:\s*"([^"]+)"', response_text)
                if match:
                    return {
                        'sentence': match.group(1),
                        'used_glosses': selected_glosses,
                        'dropped_glosses': []
                    }
            except Exception:
                pass

            return {
                'sentence': ' '.join(selected_glosses),
                'used_glosses': selected_glosses,
                'dropped_glosses': []
            }

    def _parse_rewrite_response(
        self,
        response: str,
        selected_glosses: List[str],
        running_caption: str
    ) -> Dict[str, Any]:
        """Parse Stage 2 rewrite response."""
        response_text = self._clean_json_response(response)

        try:
            result = json.loads(response_text)
            new_sentence = result.get('new_sentence', ' '.join(selected_glosses))
            full_caption = result.get('full_caption', f"{running_caption} {new_sentence}".strip())
            return {
                'new_sentence': new_sentence,
                'full_caption': full_caption,
                'used_glosses': result.get('used_glosses', selected_glosses),
                'dropped_glosses': result.get('dropped_glosses', [])
            }
        except json.JSONDecodeError:
            new_sentence = ' '.join(selected_glosses)
            return {
                'new_sentence': new_sentence,
                'full_caption': f"{running_caption} {new_sentence}".strip(),
                'used_glosses': selected_glosses,
                'dropped_glosses': []
            }

    # =========================================================================
    # FULL PIPELINE
    # =========================================================================

    def run_full_pipeline(
        self,
        glosses: List[Dict[str, Any]],
        context_sentences: Optional[List[str]] = None,
        running_caption: Optional[str] = None,
        is_reconstruction: bool = False,
        reconstruction_glosses: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Run the complete two-stage pipeline.

        Args:
            glosses: List of gloss data with top-k predictions
            context_sentences: Previous sentences for context
            running_caption: Current full caption (for rewrite mode)
            is_reconstruction: Whether to rewrite full caption
            reconstruction_glosses: All glosses in reconstruction window

        Returns:
            {
                'stage1': {...},  # Selection results
                'stage2': {...},  # Sentence results
                'final_sentence': str,
                'final_caption': str,
                'success': bool
            }
        """
        # Stage 1: Selection
        stage1_result = self.run_stage1_selection(glosses, context_sentences)
        selected_glosses = stage1_result['selections']

        # Stage 2: Sentence (or Sentence + Rewrite)
        if is_reconstruction and running_caption:
            stage2_result = self.run_stage2_sentence_rewrite(
                selected_glosses,
                running_caption,
                reconstruction_glosses
            )
            final_sentence = stage2_result.get('new_sentence', '')
            final_caption = stage2_result.get('full_caption', '')
        else:
            stage2_result = self.run_stage2_sentence(selected_glosses, context_sentences)
            final_sentence = stage2_result.get('sentence', '')
            final_caption = final_sentence

        return {
            'stage1': stage1_result,
            'stage2': stage2_result,
            'final_sentence': final_sentence,
            'final_caption': final_caption,
            'selected_glosses': selected_glosses,
            'success': stage1_result.get('success', False) and stage2_result.get('success', False)
        }

    # =========================================================================
    # UTILITIES
    # =========================================================================

    def _clean_json_response(self, response: str) -> str:
        """Clean LLM response for JSON parsing."""
        text = response.strip()

        # Remove markdown code blocks
        if text.startswith("```json"):
            text = text[7:]
        if text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]

        return text.strip()

    def _record_selection(self, glosses: List[Dict], selections: List[str]):
        """Record selection for accuracy measurement."""
        for i, (gloss_data, selection) in enumerate(zip(glosses, selections)):
            top_k = gloss_data.get('top_k', [])
            model_top1 = top_k[0].get('gloss') if top_k else gloss_data.get('gloss')
            model_conf = top_k[0].get('confidence') if top_k else gloss_data.get('confidence')

            self.selection_history.append({
                'position': i + 1,
                'model_top1': model_top1,
                'model_confidence': model_conf,
                'llm_selection': selection,
                'top_k': top_k[:3] if top_k else [],
                'selection_matches_top1': selection == model_top1
            })

    def get_selection_stats(self) -> Dict[str, Any]:
        """Get statistics about LLM selections vs model top-1."""
        if not self.selection_history:
            return {'total': 0, 'matches_top1': 0, 'differs_from_top1': 0}

        matches = sum(1 for s in self.selection_history if s['selection_matches_top1'])
        total = len(self.selection_history)

        return {
            'total': total,
            'matches_top1': matches,
            'differs_from_top1': total - matches,
            'match_rate': matches / total if total > 0 else 0,
            'change_rate': (total - matches) / total if total > 0 else 0
        }

    def clear_history(self):
        """Clear selection history."""
        self.selection_history = []
