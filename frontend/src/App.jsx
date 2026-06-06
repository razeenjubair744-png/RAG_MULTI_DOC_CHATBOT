import { useState } from 'react';
import './index.css';

function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!input.trim()) return;

    const userMsg = input.trim();
    setMessages(prev => [...prev, { text: userMsg, sender: 'user' }]);
    setInput('');
    setIsLoading(true);

    try {
      const response = await fetch('/chat/', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: userMsg, session_id: 'default' })
      });

      const data = await response.json();
      
      setMessages(prev => [...prev, { 
        text: data.answer, 
        sender: 'assistant',
        sources: data.sources 
      }]);
    } catch (error) {
      setMessages(prev => [...prev, { text: "Error connecting to server.", sender: 'assistant' }]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="app-container">
      <header className="app-header">
        <h1>RAG Chatbot</h1>
      </header>

      <div className="chat-container">
        {messages.map((msg, idx) => (
          <div key={idx} className={`message ${msg.sender}`}>
            <div className="message-content">{msg.text}</div>
            {msg.sources && msg.sources.length > 0 && (
              <div className="sources">
                <details>
                  <summary>{msg.sources.length} source(s)</summary>
                  {msg.sources.map((src, i) => (
                    <div key={i} className="source-item">
                      <strong>{src.metadata.source_file} — page {src.metadata.page_number}</strong>
                      <p>"{src.snippet}"</p>
                    </div>
                  ))}
                </details>
              </div>
            )}
          </div>
        ))}
        {isLoading && <div className="message assistant">Thinking...</div>}
      </div>

      <form className="input-form" onSubmit={handleSubmit}>
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Ask a question about your documents..."
          disabled={isLoading}
        />
        <button type="submit" disabled={isLoading || !input.trim()}>
          Send
        </button>
      </form>
    </div>
  );
}

export default App;
