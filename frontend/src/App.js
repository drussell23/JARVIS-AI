import React, { useState } from 'react';
import axios from 'axios';
import './App.css';

function App() {
  const [input, setInput] = useState('');
  const [chat, setChat] = useState([]);

  const handleSend = async () => {
    if (!input.trim()) return;

    // Append user's message to chat window
    const newChat = [...chat, { sender: 'User', message: input }];

    try {
      // Send user input to the FastAPI backend
      const response = await axios.post('http://127.0.0.1:8000/chat', {
        user_input: input,
      });
      const assistantReply = response.data.reply;
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

  return (
    <div className="App">
      <h1>ChatBot Interface</h1>
      <div className="chat-window">
        {chat.map((entry, index) => (
          <div key={index} className={`chat-entry ${entry.sender.toLowerCase()}`}>
            <strong>{entry.sender}:</strong> {entry.message}
          </div>
        ))}
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
    </div>
  );
}

export default App;
