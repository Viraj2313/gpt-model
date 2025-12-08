import { useRef, useEffect } from "react";
import { Send } from "lucide-react";

export default function InputBox({ input, setInput, onSend, isLoading }) {
  const textareaRef = useRef();

  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
      textareaRef.current.style.height = `${textareaRef.current.scrollHeight}px`;
    }
  }, [input]);

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      onSend();
    }
  };

  const handleSend = () => {
    if (input.trim() && !isLoading) onSend();
  };

  return (
    <div className="fixed bottom-0 left-0 right-0 z-40 bg-gradient-to-t from-[#faf9f5]/95 via-[#faf9f5] to-transparent pb-4 sm:pb-6 pt-12 pointer-events-none">
      <div className="max-w-4xl mx-auto px-4 sm:px-6 pointer-events-auto">
        <div className="relative bg-white border border-gray-200/80 rounded-2xl shadow-lg hover:shadow-xl transition-all duration-200 focus-within:border-gray-400 focus-within:shadow-md">
          <textarea
            ref={textareaRef}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Message Formyl GPT..."
            rows={1}
            className="w-full bg-transparent text-gray-900 placeholder-gray-500 px-4 sm:px-5 py-3.5 sm:py-4 pr-10 sm:pr-12 outline-none resize-none max-h-40 text-sm sm:text-base leading-relaxed font-light"
            disabled={isLoading}
          />
          <button
            onClick={handleSend}
            disabled={!input.trim() || isLoading}
            className="absolute right-2 sm:right-3 bottom-2 sm:bottom-3 w-8 h-8 sm:w-10 sm:h-10 bg-black text-white rounded-full flex items-center justify-center shadow-md transition-all duration-200 hover:scale-105 hover:shadow-lg disabled:bg-gray-300 disabled:cursor-not-allowed disabled:hover:scale-100"
          >
            <Send size={16} className="sm:size-18" />
          </button>
        </div>
      </div>
    </div>
  );
}
