import { Sparkles } from "lucide-react";

export default function EmptyState() {
  return (
    <div className="flex-1 flex items-center justify-center px-4 sm:px-6 py-8">
      <div className="text-center space-y-6 sm:space-y-8 animate-fadeIn">
        <div className="relative mx-auto w-20 h-20 sm:w-24 sm:h-24">
          <div className="absolute inset-0 bg-black/10 rounded-full blur-xl scale-110" />
          <div className="relative w-full h-full bg-black rounded-2xl flex items-center justify-center shadow-lg border border-gray-200/50">
            <Sparkles size={40} className="text-white sm:size-44" />
          </div>
        </div>
        <div className="space-y-2">
          <h1 className="text-3xl sm:text-4xl font-bold text-gray-900 tracking-tight">
            Formyl GPT
          </h1>
          <p className="text-gray-600 text-base sm:text-lg max-w-md mx-auto">
            How can I assist you today?
          </p>
        </div>
      </div>
    </div>
  );
}
