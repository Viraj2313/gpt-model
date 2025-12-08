import { User, Bot } from "lucide-react";

export default function Message({ role, text, isLoading }) {
  return (
    <div
      className={`w-full py-6 px-4 sm:py-7 sm:px-5 transition-all duration-300 ease-in-out opacity-0 animate-fadeIn ${
        role === "ai" ? "bg-white" : "bg-gray-50/50"
      }`}
    >
      <div className="max-w-3xl mx-auto flex gap-3 sm:gap-4 items-start">
        <div
          className={`flex-shrink-0 w-10 h-10 sm:w-11 sm:h-11 rounded-xl flex items-center justify-center text-white shadow-sm transition-transform duration-200 hover:scale-105 ${
            role === "user" ? "bg-gray-800" : "bg-black"
          }`}
        >
          {role === "user" ? <User size={18} /> : <Bot size={18} />}
        </div>

        <div className="flex-1 pt-1 min-w-0">
          {isLoading ? (
            <div className="flex gap-1.5 sm:gap-2">
              {[0, 150, 300].map((d) => (
                <div
                  key={d}
                  className="w-1.5 h-1.5 sm:w-2 sm:h-2 bg-gray-500 rounded-full animate-bounce"
                  style={{ animationDelay: `${d}ms` }}
                />
              ))}
            </div>
          ) : (
            <div className="text-gray-800 text-sm sm:text-base leading-relaxed whitespace-pre-wrap break-words font-normal">
              {text}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
