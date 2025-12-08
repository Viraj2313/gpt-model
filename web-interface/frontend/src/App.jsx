import Navbar from "./components/Navbar";
import ChatPage from "./pages/ChatPage";

export default function App() {
  return (
    <>
      <div className="fixed inset-0 bg-[#faf9f5]" />
      <div className="relative flex flex-col h-screen">
        <Navbar />
        <ChatPage />
      </div>
    </>
  );
}
