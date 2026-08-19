const CHAT_KEY = "sickleguide_chats_v5";
const ACTIVE_CHAT_KEY = `${CHAT_KEY}_active`;

try {
  const stored = JSON.parse(localStorage.getItem(CHAT_KEY) || "[]");
  const chats = Array.isArray(stored) ? stored : [];
  const freshChat = {
    id: globalThis.crypto?.randomUUID?.() || `${Date.now()}-${Math.random()}`,
    title: "New conversation",
    createdAt: Date.now(),
    updatedAt: Date.now(),
    messages: [],
  };

  localStorage.setItem(CHAT_KEY, JSON.stringify([freshChat, ...chats]));
  localStorage.setItem(ACTIVE_CHAT_KEY, freshChat.id);
} catch (error) {
  console.warn("SickleGuide session initialization failed:", error);
  localStorage.removeItem(ACTIVE_CHAT_KEY);
}
