import { useState, useRef, useEffect } from "react";
import Message from "../components/Message";
import EmptyState from "../components/EmptyState";
import InputBox from "../components/InputBox";

export default function ChatPage() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const bottomRef = useRef();

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const sendMessage = async () => {
    if (!input.trim() || isLoading) return;
    const userText = input;
    setInput("");
    setMessages((prev) => [...prev, { role: "user", text: userText }]);
    setIsLoading(true);

    try {
      const res = await fetch("http://localhost:8000/generate-stream", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt: userText }),
      });

      if (!res.ok) throw new Error("API error");

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let aiText = "";

      setMessages((prev) => [...prev, { role: "ai", text: "" }]);

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        aiText += decoder.decode(value);
        setMessages((prev) => {
          const copy = [...prev];
          copy[copy.length - 1].text = aiText;
          return copy;
        });
      }
    } catch (err) {
      console.error("Error:", err);
      setMessages((prev) => [
        ...prev,
        {
          role: "ai",
          text: "Sorry, there was an error processing your request.",
        },
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="flex-1 overflow-y-auto pt-16 sm:pt-20 pb-20 sm:pb-32 bg-[#faf9f5]">
      {messages.length === 0 ? (
        <EmptyState />
      ) : (
        <>
          {messages.map((m, i) => (
            <Message key={i} role={m.role} text={m.text} isLoading={false} />
          ))}
          {isLoading && <Message role="ai" text="" isLoading={true} />}
          <div ref={bottomRef} />
        </>
      )}
      <InputBox
        input={input}
        setInput={setInput}
        onSend={sendMessage}
        isLoading={isLoading}
      />
    </div>
  );
}
