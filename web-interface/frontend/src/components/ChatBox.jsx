import { useState, useRef, useEffect } from "react";
import { Send, Sparkles } from "lucide-react";
import Message from "./Message";

export default function ChatBox({ messages, onSend, isLoading }) {
  const [input, setInput] = useState("");
  const bottomRef = useRef();
  const textareaRef = useRef();

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
      textareaRef.current.style.height =
        textareaRef.current.scrollHeight + "px";
    }
  }, [input]);

  const send = () => {
    if (!input.trim() || isLoading) return;
    onSend(input);
    setInput("");
    if (textareaRef.current) textareaRef.current.style.height = "auto";
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      send();
    }
  };

  return (
    <div className="flex flex-col h-full max-w-4xl mx-auto w-full">
      {messages.length === 0 ? (
        <div className="flex-1 flex items-center justify-center px-4">
          <div className="text-center space-y-8 animate-fade-in">
            <div className="relative inline-block">
              <div className="absolute inset-0 bg-black/20 blur-3xl rounded-full"></div>
              <div className="relative w-24 h-24 mx-auto bg-black rounded-3xl flex items-center justify-center shadow-2xl border border-gray-800">
                <Sparkles size={44} className="text-white" />
              </div>
            </div>
            <div>
              <h2 className="text-5xl font-bold text-gray-900 mb-3">
                Formyl GPT
              </h2>
              <p className="text-gray-600 text-lg">
                How can I assist you today?
              </p>
            </div>
          </div>
        </div>
      ) : (
        <div className="flex-1 overflow-y-auto">
          {messages.map((m, i) => (
            <Message key={i} role={m.role} text={m.text} />
          ))}
          {isLoading && messages[messages.length - 1]?.role === "user" && (
            <Message role="ai" text="" isLoading={true} />
          )}
          <div ref={bottomRef} />
        </div>
      )}

      <div className="p-4 pb-8">
        <div className="relative bg-white border border-gray-300 rounded-3xl shadow-xl hover:shadow-2xl transition-all duration-300 focus-within:border-gray-600 focus-within:shadow-black/10">
          <textarea
            ref={textareaRef}
            className="w-full bg-transparent text-gray-900 p-5 pr-16 outline-none resize-none max-h-48 overflow-y-auto placeholder-gray-500 text-lg"
            placeholder="Message Formyl GPT..."
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            rows={1}
            disabled={isLoading}
          />
          <button
            onClick={send}
            disabled={!input.trim() || isLoading}
            className="absolute right-4 bottom-4 w-12 h-12 rounded-2xl bg-black text-white flex items-center justify-center disabled:bg-gray-400 disabled:cursor-not-allowed transition-all duration-300 hover:scale-110 hover:shadow-xl"
          >
            <Send size={20} />
          </button>
        </div>
      </div>
    </div>
  );
}
