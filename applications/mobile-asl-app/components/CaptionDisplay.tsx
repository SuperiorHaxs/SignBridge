import React from 'react';
import { View, Text, StyleSheet } from 'react-native';

interface CaptionDisplayProps {
  caption: string;
  style?: object;
}

export default function CaptionDisplay({ caption, style }: CaptionDisplayProps) {
  if (!caption) {
    return null;
  }

  return (
    <View style={[styles.container, style]}>
      <Text style={styles.caption}>{caption}</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    backgroundColor: 'rgba(0, 0, 0, 0.85)',
    paddingHorizontal: 20,
    paddingVertical: 15,
    borderRadius: 12,
    marginHorizontal: 10,
    marginVertical: 5,
  },
  caption: {
    color: '#fff',
    fontSize: 18,
    fontWeight: '500',
    textAlign: 'center',
    lineHeight: 26,
  },
});
