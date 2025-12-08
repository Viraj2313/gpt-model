import { Sparkles } from "lucide-react";

export default function Navbar() {
  return (
    <nav className="fixed top-0 left-0 right-0 z-50 border-b border-gray-200/80 bg-[#faf9f5]/95 backdrop-blur-xl transition-all duration-300">
      <div className="max-w-4xl mx-auto px-4 sm:px-6 py-4 flex items-center gap-3">
        <div className="w-9 h-9 bg-black rounded-lg flex items-center justify-center shadow-md transition-transform hover:scale-105">
          <Sparkles size={18} className="text-white" />
        </div>
        <h1 className="text-xl sm:text-2xl font-bold text-gray-900 tracking-tight">
          Formyl GPT
        </h1>
      </div>
    </nav>
  );
}
