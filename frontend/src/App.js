import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import './App.css';
import JarvisVoice from './components/JarvisVoice';
import VisionConnection from './components/VisionConnection';
import WorkspaceMonitor from './components/WorkspaceMonitor';
import ActionDisplay from './components/ActionDisplay';

function App() {
  const [input, setInput] = useState('');
  const [chat, setChat] = useState([]);
  const [visionData, setVisionData] = useState(null);
  const [autonomousMode, setAutonomousMode] = useState(false);
  const [visionStatus, setVisionStatus] = useState('disconnected');
  const [autonomousActions, setAutonomousActions] = useState([]);
  const visionConnectionRef = useRef(null);

  const handleSend = async () => {
    if (!input.trim()) return;

    // Append user's message to chat window
    const newChat = [...chat, { sender: 'User', message: input }];

    try {
      // Send user input to the FastAPI backend
      const response = await axios.post('http://127.0.0.1:8000/chat', {
        user_input: input,
      });
      const assistantReply = response.data?.response || response.data?.reply || 'Received.';
      newChat.push({ sender: 'Assistant', message: assistantReply });
    } catch (error) {
      console.error('Error sending message:', error);
      newChat.push({ sender: 'Assistant', message: 'Sorry, an error occurred.' });
    }

    setChat(newChat);
    setInput('');
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter') {
      handleSend();
    }
  };

  // Initialize Vision Connection
  useEffect(() => {
    if (!visionConnectionRef.current) {
      visionConnectionRef.current = new VisionConnection(
        // Callback for workspace updates
        (data) => {
          setVisionData(data);
          
          // Handle different update types
          if (data.type === 'status_update') {
            // Update vision status based on backend status
            setVisionStatus(data.status.connected ? 'connected' : 'disconnected');
            if (data.status.connected) {
              addVisionMessage('Vision system connected - monitoring active with purple indicator');
            } else {
              addVisionMessage('Vision system disconnected');
            }
          } else if (data.type === 'initial') {
            setVisionStatus('connected');
            addVisionMessage(`Vision system connected. Monitoring ${data.workspace.window_count} windows.`);
          } else if (data.notifications && data.notifications.length > 0) {
            setVisionStatus('connected');
            addVisionMessage(`Detected: ${data.notifications[0]}`);
          }
        },
        // Callback for action execution
        (data) => {
          // Handle autonomous actions from backend
          if (data.type === 'autonomous_action') {
            const newAction = {
              ...data.action,
              id: Date.now() + Math.random(),
              timestamp: new Date().toISOString(),
              status: 'pending'
            };
            setAutonomousActions(prev => [...prev, newAction]);
          } else if (data.type === 'action_executed') {
            // Update action status when executed
            setAutonomousActions(prev => 
              prev.map(a => a.id === data.action.id 
                ? {...a, status: 'executed', success: data.success} 
                : a
              )
            );
            addVisionMessage(`Action executed: ${data.action.type} on ${data.action.target}`);
          }
        }
      );
    }
    
    // Connect to vision system
    if (autonomousMode && visionConnectionRef.current) {
      visionConnectionRef.current.connect();
    }
    
    return () => {
      if (visionConnectionRef.current && !autonomousMode) {
        visionConnectionRef.current.disconnect();
      }
    };
  }, [autonomousMode]);

  const addVisionMessage = (message) => {
    setChat(prev => [...prev, { sender: 'JARVIS Vision', message }]);
  };

  const toggleAutonomousMode = () => {
    setAutonomousMode(!autonomousMode);
    if (!autonomousMode) {
      addVisionMessage('Autonomous mode activated. I can now see and respond to your workspace.');
    } else {
      addVisionMessage('Autonomous mode deactivated.');
      setVisionStatus('disconnected');
    }
  };

  return (
    <div className="App">
      <h1>J.A.R.V.I.S. Interface</h1>
      <p className="app-subtitle">Just A Rather Very Intelligent System</p>
      <div className="hud-divider" />
      
      {/* Autonomous Mode Toggle */}
      <div className="autonomous-control">
        <button 
          onClick={toggleAutonomousMode} 
          className={`autonomous-toggle ${autonomousMode ? 'active' : ''}`}
        >
          {autonomousMode ? '🤖 Autonomous Mode ON' : '👤 Manual Mode'}
        </button>
        <span className={`vision-status ${visionStatus}`}>
          Vision: {visionStatus}
        </span>
      </div>
      
      {/* Voice Control Section */}
      <JarvisVoice />
      
      <div className="hud-divider" />
      
      {/* Main Content Area */}
      <div className="main-content">
        {/* Workspace Monitor and Action Display - Show when autonomous mode is active */}
        {autonomousMode && (
          <div className="autonomous-container">
            <div className="workspace-monitor-container">
              <WorkspaceMonitor 
                visionData={visionData} 
                autonomousMode={autonomousMode}
              />
            </div>
            <ActionDisplay 
              actions={autonomousActions}
              onApprove={(action) => {
                console.log('Approved action:', action);
                // Update action status
                setAutonomousActions(prev => 
                  prev.map(a => a.id === action.id 
                    ? {...a, status: 'executed', success: true} 
                    : a
                  )
                );
                // Send approval to backend
                if (visionConnectionRef.current) {
                  visionConnectionRef.current.send({
                    type: 'execute_action',
                    action: action
                  });
                }
              }}
              onReject={(action) => {
                console.log('Rejected action:', action);
                // Remove from pending
                setAutonomousActions(prev => 
                  prev.filter(a => a.id !== action.id)
                );
              }}
            />
          </div>
        )}
        
        {/* Chat Window */}
        <div className="chat-window">
          {chat.map((entry, index) => (
            <div key={index} className={`chat-entry ${entry.sender.toLowerCase().replace(' ', '-')}`}>
              <div className="chat-bubble">
                <div className="sender-label">{entry.sender}</div>
                <div>{entry.message}</div>
              </div>
            </div>
          ))}
        </div>
      </div>
      
      <div className="chat-input">
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyPress={handleKeyPress}
          placeholder="Type your message..."
        />
        <button onClick={handleSend}>Send</button>
      </div>
      <div className="status-strip" />
    </div>
  );
}

export default App;
