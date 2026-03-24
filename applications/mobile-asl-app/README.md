# ASL Communicator - Mobile App

A React Native (Expo) mobile app for two-way communication between ASL signers and non-ASL speakers.

## Features

- **ASL Signer Mode**: Sign into the camera, see real-time captions
- **Listener Mode**: Speech-to-text for non-ASL speakers
- **Conversation Mode**: Split-screen for both participants

## Architecture

The app communicates with a local Python backend server that:
1. Extracts pose from video frames using MediaPipe
2. Predicts ASL signs using the OpenHands model
3. Constructs natural sentences using Gemini AI

```
┌────────────────────┐          ┌─────────────────────┐
│   Mobile App       │  HTTP/   │   Python Backend    │
│   (React Native)   │  WS      │   (Flask)           │
│                    │◄────────►│                     │
│   - Camera         │          │   - MediaPipe       │
│   - UI             │          │   - OpenHands Model │
│   - Speech         │          │   - Gemini AI       │
└────────────────────┘          └─────────────────────┘
```

## Prerequisites

### For the Backend Server
- Python 3.8+
- CUDA-capable GPU (optional, for faster inference)
- Gemini API key

### For the Mobile App
- Node.js 18+
- Expo CLI
- iOS device with Expo Go app (for testing)
- Or: Xcode for iOS simulator

## Setup

### 1. Backend Server Setup

```bash
# Navigate to server directory
cd applications/mobile-asl-app/server

# Create virtual environment (optional)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set Gemini API key (optional - has default)
export GEMINI_API_KEY="your-api-key"

# Start the server
python mobile_backend.py --host 0.0.0.0 --port 5000
```

The server will:
- Initialize MediaPipe for pose extraction
- Load the ASL prediction model
- Start listening on port 5000

### 2. Find Your Computer's IP Address

The mobile app needs to connect to the backend server over your local network.

**On Windows:**
```bash
ipconfig
# Look for "IPv4 Address" under your active network adapter
```

**On Mac/Linux:**
```bash
ifconfig | grep "inet "
# or
ip addr show
```

### 3. Mobile App Setup

```bash
# Navigate to mobile app directory
cd applications/mobile-asl-app

# Install dependencies
npm install

# Start Expo
npx expo start
```

### 4. Run on iOS Device

1. Install **Expo Go** app from the App Store
2. Scan the QR code shown in the terminal
3. The app will load on your device

### 5. Configure Connection

1. Open the app
2. Enter your computer's IP address (from step 2)
3. Keep port as 5000 (or change if you used a different port)
4. Tap "Connect"

## Usage

### ASL Signer Mode
1. Select "ASL Signer" from the home screen
2. Hold the "Hold to Sign" button while signing
3. Release when done - the sign will be predicted
4. After 3 signs, Gemini will construct a natural sentence

### Listener Mode
1. Select "Listener" from the home screen
2. Tap to start speech recognition
3. Your speech is transcribed to text

### Conversation Mode
1. Select "Conversation" for split-screen view
2. Toggle between ASL and Speaker modes
3. Messages appear in a chat-like interface

## API Endpoints

The backend server exposes:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/api/process-frame` | POST | Send video frame for pose extraction |
| `/api/predict-sign` | POST | Trigger sign prediction |
| `/api/add-gloss` | POST | Manually add a gloss |
| `/api/construct-sentence` | POST | Force sentence construction |
| `/api/reset` | POST | Reset session |
| `/api/status` | GET | Get session status |

## WebSocket Events

| Event | Direction | Description |
|-------|-----------|-------------|
| `frame` | Client → Server | Send video frame |
| `end_sign` | Client → Server | Signal end of sign |
| `gloss_detected` | Server → Client | New gloss detected |
| `sentence_update` | Server → Client | New sentence constructed |

## Troubleshooting

### "Connection Failed"
- Ensure the backend server is running
- Verify the IP address is correct
- Make sure both devices are on the same WiFi network
- Check if firewall is blocking port 5000

### "No pose detected"
- Ensure good lighting
- Keep your hands visible in frame
- Face the camera directly

### Slow predictions
- The first prediction may be slow (model loading)
- Subsequent predictions should be faster
- Consider using a GPU for the backend

## Development

### Modifying the App

```bash
# Start in development mode
npx expo start --dev

# Run on iOS simulator
npx expo run:ios
```

### Building for Distribution

```bash
# Create production build
npx expo build:ios
```

## File Structure

```
mobile-asl-app/
├── App.tsx                 # Main entry point
├── app.json               # Expo configuration
├── package.json           # Dependencies
├── screens/
│   ├── HomeScreen.tsx     # Server connection + mode selection
│   ├── SignerScreen.tsx   # ASL signer camera view
│   ├── ListenerScreen.tsx # Speech-to-text view
│   └── ConversationScreen.tsx # Split-screen conversation
├── components/
│   ├── CaptionDisplay.tsx # Caption overlay component
│   └── ConnectionStatus.tsx # Connection indicator
├── services/
│   ├── api.ts             # HTTP API client
│   └── socket.ts          # WebSocket client
└── server/
    ├── mobile_backend.py  # Flask backend server
    └── requirements.txt   # Python dependencies
```
