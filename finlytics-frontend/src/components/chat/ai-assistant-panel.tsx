"use client";

import { AnimatePresence, motion } from "framer-motion";
import { SendHorizontal, Sparkles } from "lucide-react";
import { useMemo, useState } from "react";

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";

interface AIAssistantPanelProps {
  open: boolean;
  onClose: () => void;
}

type ChatMessage = {
  role: "user" | "assistant";
  content: string;
};

const starter: ChatMessage[] = [
  {
    role: "assistant",
    content:
      "I am Finlytics Copilot. Ask me about risk spikes, churn strategy, or portfolio stress actions."
  }
];

function createAssistantReply(prompt: string) {
  const lower = prompt.toLowerCase();
  if (lower.includes("default")) {
    return "Default risk has a high concentration in Grade D and E cohorts. Recommend tightening underwriting filters and monitoring DTI > 35%.";
  }
  if (lower.includes("churn")) {
    return "Churn risk rises for low-engagement borrowers with repayment delays. Trigger retention campaign with dynamic refinance offers.";
  }
  if (lower.includes("anomaly") || lower.includes("suspicious")) {
    return "Current anomaly profile shows sharp score spikes around high-value transactions. Increase verification threshold for amount > $15k.";
  }
  return "Recommended next action: run Stress Testing tab with +150 bps shock and compare high-risk share before approving new exposure.";
}

export function AIAssistantPanel({ open, onClose }: AIAssistantPanelProps) {
  const [messages, setMessages] = useState<ChatMessage[]>(starter);
  const [prompt, setPrompt] = useState("");

  const placeholder = useMemo(() => "Ask: Which cohort drives today’s risk spike?", []);

  const submit = () => {
    const trimmed = prompt.trim();
    if (!trimmed) {
      return;
    }
    const userMessage: ChatMessage = { role: "user", content: trimmed };
    const assistant: ChatMessage = { role: "assistant", content: createAssistantReply(trimmed) };
    setMessages((prev) => [...prev, userMessage, assistant]);
    setPrompt("");
  };

  return (
    <AnimatePresence>
      {open ? (
        <motion.aside
          className="fixed right-0 top-0 z-50 h-screen w-full max-w-md border-l border-white/10 bg-slate-950/95 p-4 backdrop-blur-xl"
          initial={{ x: 420 }}
          animate={{ x: 0 }}
          exit={{ x: 420 }}
          transition={{ type: "spring", stiffness: 220, damping: 28 }}
        >
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-semibold text-white">Ask Finlytics</p>
              <p className="text-xs text-slate-400">AI Decision Assistant</p>
            </div>
            <Button variant="secondary" size="sm" onClick={onClose}>
              Close
            </Button>
          </div>

          <div className="mt-4 flex h-[calc(100vh-170px)] flex-col gap-2 overflow-y-auto rounded-2xl border border-white/10 bg-slate-900/40 p-3">
            {messages.map((message, index) => (
              <div
                key={index}
                className={
                  message.role === "assistant"
                    ? "max-w-[90%] rounded-xl border border-cyan-400/20 bg-cyan-500/10 p-3 text-sm text-cyan-100"
                    : "ml-auto max-w-[90%] rounded-xl border border-blue-400/20 bg-blue-500/10 p-3 text-sm text-blue-100"
                }
              >
                {message.role === "assistant" ? <Sparkles className="mb-1 h-3.5 w-3.5" /> : null}
                {message.content}
              </div>
            ))}
          </div>

          <div className="mt-3 flex items-center gap-2">
            <Input
              value={prompt}
              onChange={(event) => setPrompt(event.target.value)}
              placeholder={placeholder}
              onKeyDown={(event) => {
                if (event.key === "Enter") {
                  submit();
                }
              }}
            />
            <Button size="sm" onClick={submit}>
              <SendHorizontal className="h-4 w-4" />
            </Button>
          </div>
        </motion.aside>
      ) : null}
    </AnimatePresence>
  );
}
