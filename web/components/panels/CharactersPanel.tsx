"use client";
import { useState } from "react";
import { Card, CardTitle } from "@/components/ui/Card";
import { Button } from "@/components/ui/Button";
import { Badge } from "@/components/ui/Badge";
import { Users, Plus, Sparkles, Trash2, Copy, UserCircle } from "lucide-react";

interface Character { id: string; name: string; description: string; visual: string; }

export default function CharactersPanel() {
  const [characters, setCharacters] = useState<Character[]>([]);
  const [name, setName] = useState("");
  const [desc, setDesc] = useState("");
  const [visual, setVisual] = useState("");
  const [creating, setCreating] = useState(false);

  const addCharacter = () => {
    if (!name.trim() || !visual.trim()) return;
    setCharacters([...characters, { id: crypto.randomUUID(), name, description: desc, visual }]);
    setName(""); setDesc(""); setVisual("");
    setCreating(false);
  };

  return (
    <div>
      <div className="mb-8">
        <h1 className="text-3xl font-bold gradient-text">Character Library</h1>
        <p className="text-zinc-400 mt-1">Create persistent characters for consistent appearance across scenes</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="space-y-4">
          <Card>
            <CardTitle className="text-sm mb-3">New Character</CardTitle>
            <div className="space-y-3">
              <input value={name} onChange={(e) => setName(e.target.value)} placeholder="Character name"
                className="w-full bg-zinc-800/50 border border-zinc-700 rounded-lg px-3 py-2 text-sm text-zinc-100 placeholder-zinc-500 focus:ring-2 focus:ring-indigo-500/50" />
              <input value={desc} onChange={(e) => setDesc(e.target.value)} placeholder="Brief traits (e.g., young detective, red hair)"
                className="w-full bg-zinc-800/50 border border-zinc-700 rounded-lg px-3 py-2 text-sm text-zinc-100 placeholder-zinc-500 focus:ring-2 focus:ring-indigo-500/50" />
              <textarea value={visual} onChange={(e) => setVisual(e.target.value)}
                placeholder="Detailed visual description for consistent AI generation..."
                className="w-full h-24 bg-zinc-800/50 border border-zinc-700 rounded-lg px-3 py-2 text-sm text-zinc-100 placeholder-zinc-500 focus:ring-2 focus:ring-indigo-500/50 resize-none" />
              <div className="flex gap-2">
                <Button variant="secondary" className="flex-1"><Sparkles className="w-3.5 h-3.5 mr-1.5" /> AI Generate</Button>
                <Button className="flex-1" onClick={addCharacter} disabled={!name.trim() || !visual.trim()}>
                  <Plus className="w-3.5 h-3.5 mr-1.5" /> Save
                </Button>
              </div>
            </div>
          </Card>
          <Card>
            <h4 className="text-sm font-medium text-zinc-300 mb-2">How to Use</h4>
            <ol className="text-xs text-zinc-500 space-y-1.5 list-decimal list-inside">
              <li>Create a character with detailed visual prompt</li>
              <li>Copy the prompt and paste into Story Mode</li>
              <li>Or assign characters per scene in Story Mode</li>
              <li>The visual prompt keeps subjects consistent</li>
            </ol>
          </Card>
        </div>

        <div className="lg:col-span-2">
          {characters.length > 0 ? (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {characters.map((char) => (
                <Card key={char.id} className="group relative">
                  <div className="flex items-start gap-3">
                    <div className="w-12 h-12 rounded-full bg-indigo-600/20 flex items-center justify-center flex-shrink-0">
                      <UserCircle className="w-6 h-6 text-indigo-400" />
                    </div>
                    <div className="flex-1 min-w-0">
                      <h3 className="font-medium text-sm">{char.name}</h3>
                      {char.description && <p className="text-xs text-zinc-500 mt-0.5">{char.description}</p>}
                      <p className="text-xs text-zinc-400 mt-2 line-clamp-3 italic">{char.visual}</p>
                    </div>
                  </div>
                  <div className="flex gap-2 mt-3">
                    <Button size="sm" variant="ghost" onClick={() => navigator.clipboard.writeText(char.visual)}>
                      <Copy className="w-3 h-3 mr-1" /> Copy Prompt
                    </Button>
                    <Button size="sm" variant="danger" onClick={() => setCharacters(characters.filter((c) => c.id !== char.id))}>
                      <Trash2 className="w-3 h-3 mr-1" /> Delete
                    </Button>
                  </div>
                </Card>
              ))}
            </div>
          ) : (
            <Card className="min-h-[300px] flex items-center justify-center">
              <div className="text-center">
                <Users className="w-12 h-12 text-zinc-700 mx-auto mb-3" />
                <p className="text-zinc-500 text-sm">No characters yet</p>
                <p className="text-zinc-600 text-xs mt-1">Create one to maintain consistent subjects across scenes</p>
              </div>
            </Card>
          )}
        </div>
      </div>
    </div>
  );
}
