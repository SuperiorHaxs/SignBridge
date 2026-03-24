import React, { useState, useEffect } from 'react';
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  TextInput,
  Alert,
  ActivityIndicator,
} from 'react-native';
import { NativeStackNavigationProp } from '@react-navigation/native-stack';
import { RootStackParamList } from '../App';
import apiService from '../services/api';
import socketService from '../services/socket';

type HomeScreenProps = {
  navigation: NativeStackNavigationProp<RootStackParamList, 'Home'>;
};

export default function HomeScreen({ navigation }: HomeScreenProps) {
  const [serverIp, setServerIp] = useState('192.168.1.100');
  const [serverPort, setServerPort] = useState('5000');
  const [isConnecting, setIsConnecting] = useState(false);
  const [isConnected, setIsConnected] = useState(false);

  useEffect(() => {
    // Check connection on mount
    checkConnection();

    // Subscribe to connection changes
    const unsubscribe = socketService.onConnectionChange(setIsConnected);
    return () => unsubscribe();
  }, []);

  const checkConnection = async () => {
    const healthy = await apiService.healthCheck();
    setIsConnected(healthy);
  };

  const connectToServer = async () => {
    setIsConnecting(true);

    const url = `http://${serverIp}:${serverPort}`;
    apiService.setServerUrl(url);
    socketService.setServerUrl(url);

    try {
      // Test HTTP connection
      const healthy = await apiService.healthCheck();

      if (!healthy) {
        Alert.alert('Connection Failed', 'Could not connect to the server. Make sure the backend is running and the IP address is correct.');
        setIsConnecting(false);
        return;
      }

      // Connect WebSocket
      const connected = await socketService.connect();

      if (connected) {
        setIsConnected(true);
        Alert.alert('Connected', 'Successfully connected to the server!');
      } else {
        Alert.alert('WebSocket Error', 'HTTP connected but WebSocket failed. Real-time updates may not work.');
        setIsConnected(true); // Still allow usage
      }
    } catch (error) {
      Alert.alert('Error', 'Failed to connect: ' + String(error));
    }

    setIsConnecting(false);
  };

  const resetSession = async () => {
    const success = await apiService.resetSession();
    if (success) {
      Alert.alert('Session Reset', 'The session has been reset.');
    }
  };

  return (
    <View style={styles.container}>
      <View style={styles.header}>
        <Text style={styles.title}>ASL Communicator</Text>
        <Text style={styles.subtitle}>
          Two-way communication between ASL signers and non-ASL speakers
        </Text>
      </View>

      {/* Server Connection */}
      <View style={styles.connectionSection}>
        <Text style={styles.sectionTitle}>Server Connection</Text>

        <View style={styles.inputRow}>
          <TextInput
            style={[styles.input, styles.ipInput]}
            placeholder="Server IP"
            value={serverIp}
            onChangeText={setServerIp}
            keyboardType="numeric"
          />
          <Text style={styles.colon}>:</Text>
          <TextInput
            style={[styles.input, styles.portInput]}
            placeholder="Port"
            value={serverPort}
            onChangeText={setServerPort}
            keyboardType="numeric"
          />
        </View>

        <TouchableOpacity
          style={[styles.button, styles.connectButton]}
          onPress={connectToServer}
          disabled={isConnecting}
        >
          {isConnecting ? (
            <ActivityIndicator color="#fff" />
          ) : (
            <Text style={styles.buttonText}>
              {isConnected ? 'Reconnect' : 'Connect'}
            </Text>
          )}
        </TouchableOpacity>

        <View style={styles.statusRow}>
          <View style={[styles.statusDot, isConnected ? styles.connected : styles.disconnected]} />
          <Text style={styles.statusText}>
            {isConnected ? 'Connected' : 'Not Connected'}
          </Text>
        </View>
      </View>

      {/* Mode Selection */}
      <View style={styles.modeSection}>
        <Text style={styles.sectionTitle}>Choose Mode</Text>

        <TouchableOpacity
          style={[styles.modeButton, !isConnected && styles.disabledButton]}
          onPress={() => navigation.navigate('Signer')}
          disabled={!isConnected}
        >
          <Text style={styles.modeEmoji}>Sign to Text</Text>
          <Text style={styles.modeTitle}>ASL Signer</Text>
          <Text style={styles.modeDesc}>Sign into camera, see captions</Text>
        </TouchableOpacity>

        <TouchableOpacity
          style={[styles.modeButton, !isConnected && styles.disabledButton]}
          onPress={() => navigation.navigate('Listener')}
          disabled={!isConnected}
        >
          <Text style={styles.modeEmoji}>Speech to Text</Text>
          <Text style={styles.modeTitle}>Listener</Text>
          <Text style={styles.modeDesc}>Speak into mic, see captions</Text>
        </TouchableOpacity>

        <TouchableOpacity
          style={[styles.modeButton, styles.conversationButton, !isConnected && styles.disabledButton]}
          onPress={() => navigation.navigate('Conversation')}
          disabled={!isConnected}
        >
          <Text style={styles.modeEmoji}>Two-Way Chat</Text>
          <Text style={styles.modeTitle}>Conversation</Text>
          <Text style={styles.modeDesc}>Split screen for both users</Text>
        </TouchableOpacity>
      </View>

      {/* Reset Button */}
      {isConnected && (
        <TouchableOpacity style={styles.resetButton} onPress={resetSession}>
          <Text style={styles.resetText}>Reset Session</Text>
        </TouchableOpacity>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f5f6fa',
    padding: 20,
  },
  header: {
    alignItems: 'center',
    marginBottom: 30,
  },
  title: {
    fontSize: 28,
    fontWeight: 'bold',
    color: '#2c3e50',
  },
  subtitle: {
    fontSize: 14,
    color: '#7f8c8d',
    textAlign: 'center',
    marginTop: 8,
  },
  connectionSection: {
    backgroundColor: '#fff',
    borderRadius: 12,
    padding: 20,
    marginBottom: 20,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  sectionTitle: {
    fontSize: 16,
    fontWeight: '600',
    color: '#2c3e50',
    marginBottom: 15,
  },
  inputRow: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 15,
  },
  input: {
    backgroundColor: '#f8f9fa',
    borderRadius: 8,
    padding: 12,
    fontSize: 16,
    borderWidth: 1,
    borderColor: '#ddd',
  },
  ipInput: {
    flex: 1,
  },
  portInput: {
    width: 80,
  },
  colon: {
    fontSize: 20,
    marginHorizontal: 8,
    color: '#666',
  },
  button: {
    padding: 15,
    borderRadius: 8,
    alignItems: 'center',
  },
  connectButton: {
    backgroundColor: '#3498db',
  },
  buttonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '600',
  },
  statusRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    marginTop: 15,
  },
  statusDot: {
    width: 10,
    height: 10,
    borderRadius: 5,
    marginRight: 8,
  },
  connected: {
    backgroundColor: '#27ae60',
  },
  disconnected: {
    backgroundColor: '#e74c3c',
  },
  statusText: {
    color: '#666',
    fontSize: 14,
  },
  modeSection: {
    flex: 1,
  },
  modeButton: {
    backgroundColor: '#fff',
    borderRadius: 12,
    padding: 20,
    marginBottom: 12,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  conversationButton: {
    backgroundColor: '#2c3e50',
  },
  disabledButton: {
    opacity: 0.5,
  },
  modeEmoji: {
    fontSize: 14,
    color: '#3498db',
    fontWeight: '600',
  },
  modeTitle: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#2c3e50',
    marginTop: 5,
  },
  modeDesc: {
    fontSize: 14,
    color: '#7f8c8d',
    marginTop: 4,
  },
  resetButton: {
    padding: 15,
    alignItems: 'center',
  },
  resetText: {
    color: '#e74c3c',
    fontSize: 14,
  },
});
