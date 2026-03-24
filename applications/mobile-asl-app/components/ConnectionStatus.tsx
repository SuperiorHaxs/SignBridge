import React, { useState, useEffect } from 'react';
import { View, Text, StyleSheet } from 'react-native';
import socketService from '../services/socket';

export default function ConnectionStatus() {
  const [isConnected, setIsConnected] = useState(false);

  useEffect(() => {
    setIsConnected(socketService.isConnected());

    const unsubscribe = socketService.onConnectionChange((connected) => {
      setIsConnected(connected);
    });

    return () => unsubscribe();
  }, []);

  return (
    <View style={[styles.container, isConnected ? styles.connected : styles.disconnected]}>
      <View style={[styles.dot, isConnected ? styles.connectedDot : styles.disconnectedDot]} />
      <Text style={styles.text}>
        {isConnected ? 'Connected' : 'Disconnected'}
      </Text>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 15,
  },
  connected: {
    backgroundColor: 'rgba(39, 174, 96, 0.1)',
  },
  disconnected: {
    backgroundColor: 'rgba(231, 76, 60, 0.1)',
  },
  dot: {
    width: 8,
    height: 8,
    borderRadius: 4,
    marginRight: 6,
  },
  connectedDot: {
    backgroundColor: '#27ae60',
  },
  disconnectedDot: {
    backgroundColor: '#e74c3c',
  },
  text: {
    fontSize: 12,
    fontWeight: '500',
    color: '#666',
  },
});
