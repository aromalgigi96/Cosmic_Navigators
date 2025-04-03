import React, { useState } from 'react';
import './ChatBot.css';

const OPENAI_API_KEY = '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';
// We'll use NASA's APOD key for APOD queries, but for space debris, no key is required
const NASA_API_KEY = 'O9HtEIY3f2P5gYvSgSKNVcuVcOOTDlbeBn87etie';

const ChatBot = () => {
  const [isOpen, setIsOpen] = useState(false);
  const [messages, setMessages] = useState([
    {
      id: Date.now(),
      sender: 'bot',
      text: "Hello! I'm your NASA Assistant. Ask me anything about NASA—try 'space debris', 'APOD', or 'Mars Rover'."
    }
  ]);
  const [inputMessage, setInputMessage] = useState('');
  const [loading, setLoading] = useState(false);
  
  // We'll store the last fetched NASA data for follow-up queries
  const [lastNASAData, setLastNASAData] = useState(null);

  // Fetch APOD data from NASA
  const fetchAPOD = async () => {
    try {
      const res = await fetch(`https://api.nasa.gov/planetary/apod?api_key=${NASA_API_KEY}`);
      const data = await res.json();
      setLastNASAData({ type: 'apod', ...data });
      return {
        text: `NASA APOD: ${data.title}\n\n${data.explanation}`,
        imageUrl: data.url
      };
    } catch (error) {
      return { text: "I'm having trouble fetching NASA APOD data.", imageUrl: null };
    }
  };

  // Fetch a space debris image from NASA's Image & Video Library
  const fetchSpaceDebrisImage = async () => {
    try {
      const url = `https://images-api.nasa.gov/search?q=space+debris&media_type=image`;
      const res = await fetch(url);
      const data = await res.json();
      if (data.collection && data.collection.items && data.collection.items.length > 0) {
        const firstItem = data.collection.items[0];
        const itemData = firstItem.data && firstItem.data[0] ? firstItem.data[0] : {};
        const itemLink = firstItem.links && firstItem.links[0] ? firstItem.links[0].href : null;
       
        setLastNASAData({ type: 'debris', title: itemData.title || "Space Debris", explanation: itemData.description || "No description available.", imageUrl: itemLink });
        return {
          text: `NASA Space Debris:\nTitle: ${itemData.title || "Unknown"}\n\n${itemData.description || "No description available."}`,
          imageUrl: itemLink
        };
      } else {
        return { text: "No space debris images found.", imageUrl: null };
      }
    } catch (error) {
      return { text: "Error fetching space debris data.", imageUrl: null };
    }
  };

  // Fallback ChatGPT call
  const fetchChatGPTResponse = async (prompt) => {
    try {
      const response = await fetch("https://api.openai.com/v1/chat/completions", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "Authorization": `Bearer ${OPENAI_API_KEY}`
        },
        body: JSON.stringify({
          model: "gpt-3.5-turbo",
          messages: [
            { role: "system", content: "You are a knowledgeable NASA assistant. Answer questions about NASA in a friendly and concise manner." },
            { role: "user", content: prompt }
          ],
          temperature: 0.7,
          max_tokens: 200
        })
      });
      const data = await response.json();
      return data.choices[0].message.content;
    } catch (error) {
      return "I'm having trouble accessing the ChatGPT API right now.";
    }
  };

  // Handle user message submission
  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!inputMessage.trim()) return;
    setLoading(true);
    const userMsg = { id: Date.now(), sender: 'user', text: inputMessage };
    setMessages(prev => [...prev, userMsg]);
    const lowerInput = inputMessage.toLowerCase();
    let prompt = inputMessage;
    setInputMessage('');

    
    if (lowerInput.includes("apod") || lowerInput.includes("picture")) {
      // Fetch APOD data
      const apodData = await fetchAPOD();
      prompt += "\n\n" + apodData.text;
      
    } else if (lowerInput.includes("space debris")) {
      // Fetch a space debris image from NASA's library
      const debrisData = await fetchSpaceDebrisImage();
      prompt += "\n\n" + debrisData.text;
    } else if (lowerInput.includes("explain this") && lastNASAData) {
      prompt = `Please provide more details about "${lastNASAData.title}": ${lastNASAData.explanation}`;
    }
    
    // Get ChatGPT response using the prompt
    const botReply = await fetchChatGPTResponse(prompt);
    const botMsg = { id: Date.now() + 1, sender: 'bot', text: botReply };
    setMessages(prev => [...prev, botMsg]);
    setLoading(false);
  };

  return (
    <div className="chatbot-widget">
      <button className="chat-toggle" onClick={() => setIsOpen(!isOpen)}>
        {isOpen ? '✕' : '💬'}
      </button>
      <div className={`chat-container ${!isOpen ? 'hidden' : ''}`}>
        <div className="chat-header">
          <span>NASA Assistant</span>
          <button className="close-chat" onClick={() => setIsOpen(false)}>✕</button>
        </div>
        <div className="chat-messages">
          {messages.map((msg) => (
            <div
              key={msg.id}
              className={`message ${msg.sender === 'user' ? 'user-message' : 'bot-message'}`}
            >
              {msg.text.split('\n').map((line, i) => (
                <p key={i} style={{ margin: 0 }}>{line}</p>
              ))}
              {msg.imageUrl && (
                <img src={msg.imageUrl} alt="NASA Data" className="message-image" />
              )}
            </div>
          ))}
          {loading && <p style={{ margin: 0, color: '#c084fc' }}>Loading...</p>}
        </div>
        <form onSubmit={handleSubmit} className="chat-input-form">
          <input
            type="text"
            value={inputMessage}
            onChange={(e) => setInputMessage(e.target.value)}
            placeholder="Ask me anything about NASA..."
            className="chat-input"
          />
          <button type="submit" className="chat-submit">Send</button>
        </form>
      </div>
    </div>
  );
};

export default ChatBot;
