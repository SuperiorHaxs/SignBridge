import React from 'react';
import { NavigationContainer } from '@react-navigation/native';
import { createNativeStackNavigator } from '@react-navigation/native-stack';
import { StatusBar } from 'expo-status-bar';
import { SafeAreaProvider } from 'react-native-safe-area-context';

// Screens
import HomeScreen from './screens/HomeScreen';
import SignerScreen from './screens/SignerScreen';
import ListenerScreen from './screens/ListenerScreen';
import ConversationScreen from './screens/ConversationScreen';

// Types
export type RootStackParamList = {
  Home: undefined;
  Signer: undefined;
  Listener: undefined;
  Conversation: undefined;
};

const Stack = createNativeStackNavigator<RootStackParamList>();

export default function App() {
  return (
    <SafeAreaProvider>
      <NavigationContainer>
        <Stack.Navigator
          initialRouteName="Home"
          screenOptions={{
            headerStyle: {
              backgroundColor: '#2c3e50',
            },
            headerTintColor: '#fff',
            headerTitleStyle: {
              fontWeight: 'bold',
            },
          }}
        >
          <Stack.Screen
            name="Home"
            component={HomeScreen}
            options={{ title: 'ASL Communicator' }}
          />
          <Stack.Screen
            name="Signer"
            component={SignerScreen}
            options={{ title: 'Sign Mode' }}
          />
          <Stack.Screen
            name="Listener"
            component={ListenerScreen}
            options={{ title: 'Listen Mode' }}
          />
          <Stack.Screen
            name="Conversation"
            component={ConversationScreen}
            options={{ title: 'Conversation' }}
          />
        </Stack.Navigator>
      </NavigationContainer>
      <StatusBar style="light" />
    </SafeAreaProvider>
  );
}
