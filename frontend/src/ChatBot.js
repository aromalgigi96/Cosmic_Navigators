import React, { useState } from 'react';
import './ChatBot.css';

// NASA API key (using your provided key or DEMO_KEY for testing)
const NASA_API_KEY = 'UFYA5HcAkCUFKj6wnJVg4WJSEIlPQ0Jj7WGI1of7';

const ChatBot = () => {
  const [isOpen, setIsOpen] = useState(false);
  const [messages, setMessages] = useState([
    {
      id: Date.now(),
      sender: 'bot',
      text: "Hello! I'm your NASA Assistant. Ask me about NASA—try 'APOD', 'space debris', 'Mars Rover', or 'news'."
    }
  ]);
  const [inputMessage, setInputMessage] = useState('');
  const [loading, setLoading] = useState(false);

  // Fetch APOD data from NASA
  const fetchAPOD = async () => {
    try {
      const res = await fetch(`https://api.nasa.gov/planetary/apod?api_key=${NASA_API_KEY}`);
      const data = await res.json();
      return `NASA APOD - ${data.title}:\n${data.explanation}\nImage URL: ${data.url}`;
    } catch (error) {
      return "I'm having trouble fetching NASA APOD data.";
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
        return `NASA Space Debris - ${itemData.title || "Unknown"}:\n${itemData.description || "No description available."}`;
      } else {
        return "No space debris images found.";
      }
    } catch (error) {
      return "Error fetching space debris data.";
    }
  };

  // Fetch Mars rover photo data from NASA
  const fetchMarsRoverPhoto = async () => {
    try {
      const res = await fetch(`https://api.nasa.gov/mars-photos/api/v1/rovers/curiosity/latest_photos?api_key=${NASA_API_KEY}`);
      const data = await res.json();
      if (data.latest_photos && data.latest_photos.length > 0) {
        const photo = data.latest_photos[0];
        return `Mars Rover Photo on ${photo.earth_date}:\n${photo.img_src}`;
      }
      return "No Mars rover photos found.";
    } catch (error) {
      return "Error fetching Mars rover photos.";
    }
  };

  // Fetch NASA news using a free RSS-to-JSON converter
  const fetchNasaNews = async () => {
    try {
      // Using a free RSS-to-JSON service to convert NASA's Breaking News RSS feed into JSON
      const rssUrl = "https://api.rss2json.com/v1/api.json?rss_url=https://www.nasa.gov/rss/dyn/breaking_news.rss";
      const res = await fetch(rssUrl);
      const data = await res.json();
      if (data.items && data.items.length > 0) {
        const topItem = data.items[0];
        return `Latest NASA News: ${topItem.title}\nRead more: ${topItem.link}`;
      } else {
        return "No recent NASA news found.";
      }
    } catch (error) {
      return "Error fetching NASA news.";
    }
  };

  // Handle user message submission with flexible keyword matching
  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!inputMessage.trim()) return;
    setLoading(true);

    // Add the user's message
    const userMsg = { id: Date.now(), sender: 'user', text: inputMessage };
    setMessages(prev => [...prev, userMsg]);
    
    // Convert input to lowercase and remove punctuation for robust matching
    let lowerInput = inputMessage.toLowerCase();
    lowerInput = lowerInput.replace(/[^a-z0-9 ]/g, ' ');

    let botResponse = "";
    if (lowerInput.includes("apod") || lowerInput.includes("picture of the day")) {
      botResponse = await fetchAPOD();
    } else if (lowerInput.includes("space debris")) {
      botResponse = await fetchSpaceDebrisImage();
    } else if (lowerInput.includes("mars rover") || lowerInput.includes("mars")) {
      botResponse = await fetchMarsRoverPhoto();
    } else if (lowerInput.includes("news")) {
      botResponse = await fetchNasaNews();
    } else {
      botResponse = "I'm sorry, I can only answer questions related to NASA, such as APOD, space debris, Mars Rover, or news.";
    }

    // Add the bot's response to the conversation
    const botMsg = { id: Date.now() + 1, sender: 'bot', text: botResponse };
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
            <div key={msg.id} className={`message ${msg.sender === 'user' ? 'user-message' : 'bot-message'}`}>
              {msg.text.split('\n').map((line, i) => (
                <p key={i} style={{ margin: 0 }}>{line}</p>
              ))}
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
