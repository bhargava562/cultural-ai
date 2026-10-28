import { useEffect, useRef, useState } from 'react'
import type { FormEvent } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import {
  ArrowUp,
  Bot,
  Check,
  ChevronDown,
  Copy,
  Menu,
  MessageSquare,
  Mic,
  MicOff,
  MonitorSpeaker,
  Plus,
  RotateCcw,
  Sparkles,
  Square,
  UserRound,
  Volume2,
  VolumeX,
  X,
} from 'lucide-react'
import { KioskAvatar } from './KioskAvatar'
import type { AvatarState } from './KioskAvatar'
import './App.css'

type Role = 'user' | 'assistant'
type Language = 'English' | 'Tamil' | 'Tanglish' | 'Hinglish'
type Message = {
  id: number
  role: Role
  content: string
  spokenText?: string
  language?: Language
  error?: boolean
}

const MODEL = import.meta.env.VITE_GROQ_MODEL || 'openai/gpt-oss-120b'
const MEDIATOR_API_URL = import.meta.env.VITE_MEDIATOR_API_URL || 'http://127.0.0.1:8000'

const starterPrompts = [
  'How should I invite my elder relatives for Pongal?',
  'En thangachi payanuku kaadhu kuthu function, thaai maaman seer epdi pannanum?',
  'Moi virundhu la epdi participate panradhu?',
]

function stripMarkdownForSpeech(md: string): string {
  return md
    .replace(/```[\s\S]*?```/g, '')
    .replace(/^#{1,6}\s*/gm, '')
    .replace(/\*\*([^*]+)\*\*/g, '$1')
    .replace(/\*([^*]+)\*/g, '$1')
    .replace(/^\s*[-*•]\s+/gm, '')
    .replace(/^\s*\d+\.\s+/gm, '')
    .replace(/[🏛️✅🌿✨🎯💡]/g, '')
    .replace(/\s+/g, ' ')
    .trim()
    .split(/(?<=[.!?])\s+/)
    .slice(0, 5)
    .join(' ')
}

function detectLanguage(text: string): Language {
  const tamilCount = (text.match(/[\u0B80-\u0BFF]/g) || []).length
  const letters = (text.match(/[\p{L}]/gu) || []).length || 1
  if (tamilCount / letters > 0.3) return 'Tamil'
  const normalized = text.toLowerCase()
  const tanglishWords =
    normalized.match(
      /\b(machi|macha|mapla|naina|thambi|akka|anna|da|di|pa|vanakkam|vaanga|ponga|sollunga|eppadi|epdi|irukinga|romba|nalla|konjam|ippo|enna|yaaru|saaptiya|nanba|saptiya|sowkiyama|enga|yenga|oorla|kalyanam|seer|maaman|moi|virundhu|kovil|function)\b/g
    ) || []
  const hinglishWords =
    normalized.match(
      /\b(arey|arre|bhai|behen|aap|aapka|kaise|hai|hain|haan|kya|kyun|main|mujhe|mera|tum|nahi|bahut|achha|theek|yaar|dost|batao|bolo|chalo|namaste|shukriya|ji)\b/g
    ) || []
  if (hinglishWords.length > tanglishWords.length) return 'Hinglish'
  return tanglishWords.length > 0 ? 'Tanglish' : 'English'
}

function App() {
  const [viewMode, setViewMode] = useState<'chat' | 'kiosk'>('chat')
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [sidebarOpen, setSidebarOpen] = useState(false)
  const [copiedId, setCopiedId] = useState<number | null>(null)

  // Kiosk Voice, Avatar & Subtitle States
  const [avatarState, setAvatarState] = useState<AvatarState>('idle')
  const [isListening, setIsListening] = useState(false)
  const [voiceLocale, setVoiceLocale] = useState<'en-IN' | 'ta-IN'>('en-IN')
  const [autoSpeak, setAutoSpeak] = useState(true)
  const [subtitleText, setSubtitleText] = useState('Vanakkam!')
  const [currentWord, setCurrentWord] = useState('')
  const [spokenCharIndex, setSpokenCharIndex] = useState<number>(0)
  const [showFullDetails, setShowFullDetails] = useState(false)

  const conversationRef = useRef<HTMLDivElement>(null)
  const recognitionRef = useRef<any>(null)
  const wordTimerRef = useRef<number | null>(null)

  // Isolated auto-scroll when new messages arrive
  useEffect(() => {
    if (messages.length > 0 && conversationRef.current) {
      conversationRef.current.scrollTo({
        top: conversationRef.current.scrollHeight,
        behavior: 'smooth',
      })
    }
  }, [messages, isLoading])

  // Cleanup speech synthesis and timers on unmount
  useEffect(() => {
    return () => {
      window.speechSynthesis?.cancel()
      if (wordTimerRef.current) window.clearInterval(wordTimerRef.current)
    }
  }, [])

  const stopSpeaking = () => {
    window.speechSynthesis?.cancel()
    if (wordTimerRef.current) {
      window.clearInterval(wordTimerRef.current)
      wordTimerRef.current = null
    }
    setCurrentWord('')
    setAvatarState((prev) => (prev === 'speaking' ? 'idle' : prev))
  }

  const speakResponse = (textToSpeak: string, lang?: Language) => {
    if (!('speechSynthesis' in window)) return
    stopSpeaking()

    const cleanScript = stripMarkdownForSpeech(textToSpeak)
    if (!cleanScript) return

    setSubtitleText(cleanScript)
    setSpokenCharIndex(0)
    setAvatarState('speaking')

    const utterance = new SpeechSynthesisUtterance(cleanScript)
    const targetLang = lang === 'Tamil' || voiceLocale === 'ta-IN' ? 'ta-IN' : 'en-IN'
    utterance.lang = targetLang
    utterance.rate = 0.96
    utterance.pitch = 1.0

    const voices = window.speechSynthesis.getVoices()
    const preferredVoice =
      voices.find((v) => v.lang === targetLang) ||
      voices.find((v) => v.lang.startsWith(targetLang.split('-')[0])) ||
      voices.find((v) => v.lang.includes('IN'))
    if (preferredVoice) utterance.voice = preferredVoice

    const words = cleanScript.split(/\s+/)
    let boundaryFired = false

    utterance.onboundary = (event: SpeechSynthesisEvent) => {
      boundaryFired = true
      const idx = event.charIndex || 0
      setSpokenCharIndex(idx)
      const remaining = cleanScript.slice(idx)
      const activeWord = remaining.split(/\s+/)[0] || ''
      setCurrentWord(activeWord)
    }

    let wordIdx = 0
    wordTimerRef.current = window.setInterval(() => {
      if (!boundaryFired && wordIdx < words.length) {
        const w = words[wordIdx]
        setCurrentWord(w)
        const approxChars = words.slice(0, wordIdx + 1).join(' ').length
        setSpokenCharIndex(approxChars)
        wordIdx++
      }
    }, 310)

    utterance.onend = () => {
      if (wordTimerRef.current) window.clearInterval(wordTimerRef.current)
      setCurrentWord('')
      setSpokenCharIndex(cleanScript.length)
      setAvatarState('idle')
    }

    utterance.onerror = () => {
      if (wordTimerRef.current) window.clearInterval(wordTimerRef.current)
      setCurrentWord('')
      setAvatarState('idle')
    }

    window.speechSynthesis.speak(utterance)
  }

  const toggleVoiceInput = () => {
    const SpeechRec = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition
    if (!SpeechRec) {
      setSubtitleText('Voice input requires Chrome, Edge, or Safari with Web Speech API support.')
      return
    }

    if (isListening) {
      recognitionRef.current?.stop()
      setIsListening(false)
      setAvatarState('idle')
      return
    }

    stopSpeaking()
    const recognition = new SpeechRec()
    recognition.lang = voiceLocale
    recognition.continuous = false
    recognition.interimResults = true

    recognition.onstart = () => {
      setIsListening(true)
      setAvatarState('listening')
      setSubtitleText('Listening... Speak your cultural question now.')
    }

    recognition.onresult = (event: any) => {
      let transcript = ''
      for (let i = 0; i < event.results.length; i++) {
        transcript += event.results[i][0].transcript
      }
      setInput(transcript)
      setSubtitleText(`"${transcript}"`)

      if (event.results[0]?.isFinal) {
        setIsListening(false)
        void submitQuery(transcript)
      }
    }

    recognition.onerror = () => {
      setIsListening(false)
      setAvatarState('idle')
    }

    recognition.onend = () => {
      setIsListening(false)
    }

    recognitionRef.current = recognition
    recognition.start()
  }

  const submitQuery = async (rawQuery: string) => {
    const query = rawQuery.trim()
    if (!query || isLoading) return

    stopSpeaking()
    const language = detectLanguage(query)
    const userMessage: Message = { id: Date.now(), role: 'user', content: query, language }
    setMessages((current) => [...current, userMessage])
    setInput('')
    setIsLoading(true)
    setAvatarState('thinking')
    setSubtitleText('Consulting traditions and preparing guidance...')

    try {
      // Query FastAPI Cultural Alignment Mediator
      const mediatorResponse = await fetch(`${MEDIATOR_API_URL}/api/mediate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query, model: MODEL }),
      })

      if (!mediatorResponse.ok) {
        const errJson = await mediatorResponse.json().catch(() => null)
        const detail = errJson?.detail || `Backend returned HTTP status ${mediatorResponse.status}`
        throw new Error(detail)
      }

      const mediated = await mediatorResponse.json()
      const answer = mediated.conversational_response || mediated.mediated_response || ''
      const spoken = mediated.spoken_response || stripMarkdownForSpeech(answer)

      const assistantMsg: Message = {
        id: Date.now() + 1,
        role: 'assistant',
        language,
        content: answer,
        spokenText: spoken,
      }
      setMessages((current) => [...current, assistantMsg])
      setSubtitleText(spoken)

      if (viewMode === 'kiosk' && autoSpeak) {
        speakResponse(spoken, language)
      } else {
        setAvatarState('idle')
      }
    } catch (error) {
      const errMsg = error instanceof Error ? error.message : 'Something went wrong.'
      setMessages((current) => [
        ...current,
        { id: Date.now() + 1, role: 'assistant', error: true, content: errMsg },
      ])
      setSubtitleText(errMsg)
      setAvatarState('idle')
    } finally {
      setIsLoading(false)
    }
  }

  const sendMessage = async (event?: FormEvent) => {
    event?.preventDefault()
    await submitQuery(input)
  }

  const copyMessage = async (message: Message) => {
    await navigator.clipboard.writeText(message.content)
    setCopiedId(message.id)
    window.setTimeout(() => setCopiedId(null), 1600)
  }

  const clearChat = () => {
    stopSpeaking()
    setMessages([])
    setSubtitleText('Vanakkam!')
    setSidebarOpen(false)
  }

  const latestAssistantMessage = [...messages].reverse().find((m) => m.role === 'assistant' && !m.error)
  const latestUserMessage = [...messages].reverse().find((m) => m.role === 'user')

  return (
    <main className="app-shell">
      <div className="ambient ambient-one" />
      <div className="ambient ambient-two" />

      <aside className={`sidebar ${sidebarOpen ? 'sidebar-open' : ''}`}>
        <div className="sidebar-brand">
          <div className="brand-mark">
            <Sparkles size={16} />
          </div>
          <span>nalla.ai</span>
          <button className="icon-button mobile-close" onClick={() => setSidebarOpen(false)} aria-label="Close menu">
            <X size={18} />
          </button>
        </div>

        <button className="new-chat" onClick={clearChat}>
          <Plus size={16} /> New chat <span>⌘ K</span>
        </button>

        <div className="sidebar-section">
          <span>Recent</span>
          <button className="history-item active" onClick={clearChat}>
            <MessageSquare size={14} /> New conversation
          </button>
        </div>

        <div className="sidebar-bottom">
          <div className="sidebar-status">
            <span className="status-dot" />
            <span>
              <strong>Alignment layer</strong>
              <small>{viewMode === 'kiosk' ? 'Kiosk Active' : 'Ready'}</small>
            </span>
          </div>
          <div className="model-chip">
            <span>MODEL</span>
            <strong>{MODEL.replace('openai/', '')}</strong>
            <ChevronDown size={14} />
          </div>
        </div>
      </aside>

      {sidebarOpen && (
        <button className="sidebar-scrim" onClick={() => setSidebarOpen(false)} aria-label="Close navigation" />
      )}

      <section className="chat-panel">
        <header className="topbar">
          <button className="icon-button menu-button" onClick={() => setSidebarOpen(true)} aria-label="Open menu">
            <Menu size={20} />
          </button>
          <div className="mobile-brand">
            <div className="brand-mark">
              <Sparkles size={15} />
            </div>
            <strong>nalla.ai</strong>
          </div>
          <div className="topbar-right">
            <button
              type="button"
              className={`mode-switch-btn ${viewMode === 'kiosk' ? 'kiosk-active' : ''}`}
              onClick={() => {
                if (viewMode === 'kiosk') {
                  stopSpeaking()
                  setViewMode('chat')
                } else {
                  setViewMode('kiosk')
                }
              }}
            >
              {viewMode === 'kiosk' ? (
                <>
                  <MessageSquare size={14} />
                  <span>Switch to Chat</span>
                </>
              ) : (
                <>
                  <MonitorSpeaker size={14} />
                  <span>Enter Kiosk Mode</span>
                </>
              )}
            </button>
            <span className="live-pill">
              <span className="status-dot" /> Live
            </span>
            <button className="avatar" aria-label="Profile">
              <UserRound size={16} />
            </button>
          </div>
        </header>

        {viewMode === 'kiosk' ? (
          /* ================= INTERACTIVE KIOSK MODE ================= */
          <div className="kiosk-container">
            <div className="kiosk-toolbar">
              <div className="kiosk-lang-pills">
                <button
                  type="button"
                  className={voiceLocale === 'en-IN' ? 'active' : ''}
                  onClick={() => setVoiceLocale('en-IN')}
                >
                  Tanglish / English (IN)
                </button>
                <button
                  type="button"
                  className={voiceLocale === 'ta-IN' ? 'active' : ''}
                  onClick={() => setVoiceLocale('ta-IN')}
                >
                  தமிழ் (ta-IN)
                </button>
              </div>

              <div className="kiosk-audio-controls">
                <button
                  type="button"
                  className="kiosk-icon-pill"
                  onClick={() => {
                    const next = !autoSpeak
                    setAutoSpeak(next)
                    if (!next) stopSpeaking()
                  }}
                >
                  {autoSpeak ? <Volume2 size={14} /> : <VolumeX size={14} />}
                  <span>{autoSpeak ? 'Audio ON' : 'Audio Muted'}</span>
                </button>

                {avatarState === 'speaking' ? (
                  <button type="button" className="kiosk-icon-pill stop" onClick={stopSpeaking}>
                    <Square size={13} />
                    <span>Stop</span>
                  </button>
                ) : (
                  latestAssistantMessage && (
                    <button
                      type="button"
                      className="kiosk-icon-pill"
                      onClick={() =>
                        speakResponse(
                          latestAssistantMessage.spokenText || latestAssistantMessage.content,
                          latestAssistantMessage.language
                        )
                      }
                    >
                      <RotateCcw size={13} />
                      <span>Replay</span>
                    </button>
                  )
                )}
              </div>
            </div>

            {latestUserMessage && (
              <div className="kiosk-user-prompt-banner">
                <span>Query:</span> “{latestUserMessage.content}”
              </div>
            )}

            {/* Animated Speaking Character Stage */}
            <KioskAvatar state={avatarState} currentWord={currentWord} />

            {/* Live Subtitle Teleprompter Box */}
            <div className="kiosk-subtitle-box" aria-live="polite">
              <div className="subtitle-label">LIVE SUBTITLES</div>
              <p className="subtitle-text">
                {avatarState === 'speaking' && spokenCharIndex > 0 ? (
                  <>
                    <span className="subtitle-spoken">{subtitleText.slice(0, spokenCharIndex)}</span>
                    <span className="subtitle-upcoming">{subtitleText.slice(spokenCharIndex)}</span>
                  </>
                ) : (
                  subtitleText
                )}
              </p>

              {latestAssistantMessage && (
                <div className="kiosk-details-toggle">
                  <button type="button" onClick={() => setShowFullDetails((v) => !v)}>
                    {showFullDetails
                      ? 'Hide Full Cultural Process Steps ▲'
                      : 'View Full Cultural Process Steps ▼'}
                  </button>
                  {showFullDetails && (
                    <div className="kiosk-full-markdown">
                      <ReactMarkdown remarkPlugins={[remarkGfm]}>
                        {latestAssistantMessage.content}
                      </ReactMarkdown>
                    </div>
                  )}
                </div>
              )}
            </div>



            {/* Dual Voice + Word Input Dock */}
            <div className="kiosk-input-dock">
              <button
                type="button"
                className={`kiosk-mic-btn ${isListening ? 'listening' : ''}`}
                onClick={toggleVoiceInput}
                aria-label={isListening ? 'Stop voice input' : 'Start voice input'}
              >
                {isListening ? <MicOff size={20} /> : <Mic size={20} />}
                <span>{isListening ? 'Listening...' : 'Tap to Speak'}</span>
              </button>

              <form className="kiosk-text-form" onSubmit={sendMessage}>
                <input
                  type="text"
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  placeholder="Ask in Tanglish, Tamil, or English..."
                  aria-label="Kiosk text input"
                />
                <button type="submit" className="send-button" disabled={!input.trim() || isLoading}>
                  <ArrowUp size={18} />
                </button>
              </form>
            </div>
          </div>
        ) : (
          /* ================= STANDARD CHATBOT MODE (ChatGPT UI) ================= */
          <>
            <div className="conversation" ref={conversationRef}>
              <div className="conversation-inner">
                {messages.length === 0 ? (
                  <div className="empty-chat-hero">
                    <h2>What cultural decision can I help you with today?</h2>
                    <div className="starter-grid">
                      {starterPrompts.map((prompt) => (
                        <button key={prompt} onClick={() => void submitQuery(prompt)}>
                          <span>{prompt}</span>
                          <ArrowUp size={14} />
                        </button>
                      ))}
                    </div>
                  </div>
                ) : (
                  <div className="messages" aria-live="polite">
                    {messages.map((message) => (
                      <article className={`message-row ${message.role}`} key={message.id}>
                        <div className={`message-avatar ${message.role}`}>
                          {message.role === 'assistant' ? <Bot size={17} /> : <UserRound size={16} />}
                        </div>
                        <div className="message-content">
                          <div className="message-meta">
                            <span>{message.role === 'assistant' ? 'nalla.ai' : 'You'}</span>
                            {message.language && <span className="language-tag">{message.language}</span>}
                          </div>
                          <div className={`message-bubble ${message.error ? 'error-bubble' : ''}`}>
                            {message.role === 'assistant' ? (
                              <ReactMarkdown remarkPlugins={[remarkGfm]}>{message.content}</ReactMarkdown>
                            ) : (
                              <p>{message.content}</p>
                            )}
                          </div>
                          {message.role === 'assistant' && !message.error && (
                            <div className="message-actions">
                              <button onClick={() => copyMessage(message)}>
                                {copiedId === message.id ? <Check size={13} /> : <Copy size={13} />}{' '}
                                {copiedId === message.id ? 'Copied' : 'Copy'}
                              </button>
                              <button
                                onClick={() => {
                                  setViewMode('kiosk')
                                  speakResponse(message.spokenText || message.content, message.language)
                                }}
                              >
                                <Volume2 size={13} /> Speak in Kiosk
                              </button>
                            </div>
                          )}
                        </div>
                      </article>
                    ))}

                    {isLoading && (
                      <article className="message-row assistant">
                        <div className="message-avatar assistant">
                          <Bot size={17} />
                        </div>
                        <div className="message-content">
                          <div className="message-meta">
                            <span>nalla.ai</span>
                            <span className="language-tag">thinking</span>
                          </div>
                          <div className="message-bubble loading-bubble">
                            <span />
                            <span />
                            <span />
                          </div>
                        </div>
                      </article>
                    )}
                  </div>
                )}
              </div>
            </div>

            <div className="composer-wrap">
              <form className="composer" onSubmit={sendMessage}>
                <textarea
                  value={input}
                  onChange={(event) => setInput(event.target.value)}
                  onKeyDown={(event) => {
                    if (event.key === 'Enter' && !event.shiftKey) {
                      event.preventDefault()
                      void sendMessage()
                    }
                  }}
                  placeholder="Ask in Tamil, Tanglish, Hinglish, or English..."
                  rows={1}
                  aria-label="Message"
                />
                <button className="send-button" type="submit" disabled={!input.trim() || isLoading} aria-label="Send message">
                  <ArrowUp size={18} />
                </button>
              </form>
              <p className="composer-note">
                Enter to send <span>·</span> Shift + Enter for new line
              </p>
            </div>
          </>
        )}

        <footer className="app-footer">
          <span>nalla.ai</span>
          <span>Cultural Decision Assistant</span>
        </footer>
      </section>
    </main>
  )
}

export default App
