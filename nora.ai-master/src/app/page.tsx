"use client"
// pages/index.js
import React, { useState } from 'react';

const ChatPage = () => {
  const [inputText, setInputText] = useState('');
  const [chatResponses, setChatResponses] = useState([]);

  const handleInputChange = (e) => {
    setInputText(e.target.value);
  };

  const handleSubmit = (e) => {
    e.preventDefault();

   
    setInputText("bye");
    const response = `${inputText}`;

    setChatResponses((prevResponses) => [...prevResponses, response]);

    fetch('http://localhost:5000/noraAi', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ message: response }),
    }).then(response => response.json())
    .then(data => {console.log(data);
      setChatResponses((prevResponses) => [...prevResponses, data.prediction]);
    })

   
    

    setInputText('');
  };

  return (
   
    <div style={{height:'100vh', margin: 'auto', padding_top :'0px', padding_left : '100px', textAlign: 'center', fontFamily: 'Astro Futuristic Font', color: '#41B3A3', backgroundColor: '#EAE7DC' }}>
      
      <h1 style={{ color: '#E85A4F' }}>NORA .AI</h1>
      <div
        style={{
          border: '2px solid #4CAF50',
          borderRadius: '8px',
          padding: '20px',
          marginBottom: '20px',
          minHeight: '150px',
          backgroundColor: '#edf2f4',
        }}
      >
        
        {chatResponses.map((response, index) => (
          <div key={index} style={{ marginBottom: '10px', textAlign: 'left', color: '#2D3B2D' }}>{response}</div>
        ))}
      </div>
      <form onSubmit={handleSubmit}>
        <input
          type="text"
          value={inputText}
          onChange={handleInputChange}
          placeholder="Enter the Query..."
          style={{ padding: '10px', marginRight: '10px', borderRadius: '4px', border: '1px solid #4CAF50', color: 'white' }}
        />
        <button type="submit" style={{ padding: '10px', borderRadius: '4px', backgroundColor: '#4CAF50', color: 'white', border: 'none' }}>Submit</button>
      </form>
    </div>
  );
};

export default ChatPage;


