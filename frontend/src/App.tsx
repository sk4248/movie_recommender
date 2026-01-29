import { useState } from "react";

type Rec = { title: string; score: number; explanation: string };

export default function App() {
  const [input, setInput] = useState("Inception, Interstellar");
  const [recs, setRecs] = useState<Rec[]>([]);
  const [loading, setLoading] = useState(false);

  async function fetchRecs() {
    setLoading(true);
    try {
      const movies = input
        .split(",")
        .map((s) => s.trim())
        .filter(Boolean);

      const res = await fetch("http://localhost:8000/recommend", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ movies, n: 10, method: "item_cf" }),
      });

      const data = await res.json();
      setRecs(data.recommendations ?? []);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div style={{ padding: 24, fontFamily: "system-ui" }}>
      <h1>Movie Recommender</h1>

      <p>Enter one or more movies (comma separated):</p>
      <input
        style={{ width: "100%", padding: 12, fontSize: 16 }}
        value={input}
        onChange={(e) => setInput(e.target.value)}
      />
      <button style={{ marginTop: 12, padding: 12 }} onClick={fetchRecs} disabled={loading}>
        {loading ? "Recommending..." : "Recommend"}
      </button>

      <ul style={{ marginTop: 20 }}>
        {recs.map((r, i) => (
          <li key={i} style={{ marginBottom: 12 }}>
            <b>{r.title}</b>
            <div style={{ opacity: 0.8 }}>{r.explanation}</div>
          </li>
        ))}
      </ul>
    </div>
  );
}
