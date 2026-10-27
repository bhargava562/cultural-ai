import { useEffect, useState } from 'react'

export type AvatarState = 'idle' | 'listening' | 'thinking' | 'speaking'
export type Viseme = 'rest' | 'A' | 'E' | 'I' | 'O' | 'U' | 'MBP'

interface KioskAvatarProps {
  state: AvatarState
  currentWord: string
}

export function wordToViseme(word: string, tick: number): Viseme {
  if (!word) return 'rest'
  const clean = word.toLowerCase().replace(/[^a-z\u0B80-\u0BFF]/g, '')
  if (!clean) return 'rest'
  const char = clean[tick % clean.length] || clean[0]
  if ('mbpமபவ'.includes(char)) return 'MBP'
  if ('oauஒஓஉஊ'.includes(char)) return 'O'
  if ('eaiஎஏஐஇஈ'.includes(char)) return 'E'
  if ('uஉஊ'.includes(char)) return 'U'
  return tick % 2 === 0 ? 'A' : 'I'
}

export function KioskAvatar({ state, currentWord }: KioskAvatarProps) {
  const [blink, setBlink] = useState(false)
  const [tick, setTick] = useState(0)

  // Automatic natural eye blinking
  useEffect(() => {
    const blinkInterval = window.setInterval(() => {
      setBlink(true)
      window.setTimeout(() => setBlink(false), 140)
    }, 3400)
    return () => window.clearInterval(blinkInterval)
  }, [])

  // Lip-sync viseme oscillator while speaking
  useEffect(() => {
    if (state !== 'speaking') {
      setTick(0)
      return
    }
    const lipInterval = window.setInterval(() => {
      setTick((t) => t + 1)
    }, 110)
    return () => window.clearInterval(lipInterval)
  }, [state])

  const viseme: Viseme = state === 'speaking' ? wordToViseme(currentWord, tick) : 'rest'

  const renderMouth = () => {
    if (state === 'listening') {
      return <path d="M 88 140 Q 100 146 112 140" stroke="#3a2314" strokeWidth="3.2" fill="none" strokeLinecap="round" />
    }
    if (state === 'thinking') {
      return <ellipse cx="103" cy="141" rx="6" ry="3.5" fill="#3a2314" />
    }
    switch (viseme) {
      case 'A':
        return (
          <g>
            <path d="M 85 136 Q 100 156 115 136 Z" fill="#5c1d1d" stroke="#3a2314" strokeWidth="2.5" />
            <path d="M 89 137 L 111 137" stroke="#ffffff" strokeWidth="2.5" strokeLinecap="round" />
            <ellipse cx="100" cy="146" rx="7" ry="4" fill="#d95b5b" />
          </g>
        )
      case 'O':
      case 'U':
        return (
          <g>
            <ellipse cx="100" cy="141" rx="8.5" ry="9.5" fill="#5c1d1d" stroke="#3a2314" strokeWidth="2.5" />
            <ellipse cx="100" cy="145" rx="5" ry="3" fill="#d95b5b" />
          </g>
        )
      case 'E':
      case 'I':
        return (
          <g>
            <path d="M 84 138 Q 100 150 116 138 Q 100 135 84 138 Z" fill="#5c1d1d" stroke="#3a2314" strokeWidth="2.5" />
            <path d="M 88 139 L 112 139" stroke="#ffffff" strokeWidth="2.5" strokeLinecap="round" />
          </g>
        )
      case 'MBP':
        return <line x1="86" y1="140" x2="114" y2="140" stroke="#3a2314" strokeWidth="3.8" strokeLinecap="round" />
      case 'rest':
      default:
        return <path d="M 86 138 Q 100 148 114 138" stroke="#3a2314" strokeWidth="3.2" fill="none" strokeLinecap="round" />
    }
  }

  return (
    <div className={`kiosk-avatar-stage state-${state}`}>
      <div className="kiosk-aura-ring ring-1" />
      <div className="kiosk-aura-ring ring-2" />
      <div className="kiosk-aura-ring ring-3" />

      <svg viewBox="0 0 200 230" className="kiosk-character-svg" role="img" aria-label="Cultural Kiosk Avatar">
        <defs>
          <linearGradient id="skinGrad" x1="0%" y1="0%" x2="0%" y2="100%">
            <stop offset="0%" stopColor="#e0a370" />
            <stop offset="100%" stopColor="#c6824c" />
          </linearGradient>
          <linearGradient id="kurtaGrad" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stopColor="#f7f2e7" />
            <stop offset="100%" stopColor="#e5dcc5" />
          </linearGradient>
          <linearGradient id="angavastramGrad" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%" stopColor="#c9e86a" />
            <stop offset="100%" stopColor="#8fb83b" />
          </linearGradient>
        </defs>

        {/* Torso & Traditional Veshti/Kurta */}
        <g className="avatar-torso">
          <path d="M 40 225 C 42 175, 158 175, 160 225 Z" fill="url(#kurtaGrad)" stroke="#d4c5a1" strokeWidth="2" />
          {/* Golden-Lime Angavastram (Shawl) */}
          <path d="M 52 182 Q 75 215 95 225 L 112 225 Q 88 202 68 175 Z" fill="url(#angavastramGrad)" opacity="0.9" />
          <path d="M 148 182 Q 125 215 105 225 L 122 225 Q 138 202 132 175 Z" fill="#d4af37" opacity="0.75" />
          {/* Neck */}
          <rect x="86" y="152" width="28" height="28" rx="10" fill="url(#skinGrad)" />
        </g>

        {/* Animated Head Group */}
        <g className="avatar-head">
          {/* Ears */}
          <ellipse cx="48" cy="112" rx="8" ry="14" fill="url(#skinGrad)" />
          <ellipse cx="152" cy="112" rx="8" ry="14" fill="url(#skinGrad)" />

          {/* Face Base */}
          <rect x="52" y="55" width="96" height="106" rx="44" fill="url(#skinGrad)" />

          {/* Hair & Crown */}
          <path d="M 48 92 C 44 48, 156 48, 152 92 C 146 68, 125 60, 100 60 C 75 60, 54 68, 48 92 Z" fill="#1f1c18" />

          {/* Traditional Vibhuti & Kumkumam Tilak on Forehead */}
          <line x1="88" y1="80" x2="112" y2="80" stroke="#f5f2eb" strokeWidth="2.2" strokeLinecap="round" opacity="0.85" />
          <line x1="91" y1="84" x2="109" y2="84" stroke="#f5f2eb" strokeWidth="2.2" strokeLinecap="round" opacity="0.85" />
          <circle cx="100" cy="82" r="3.2" fill="#d93829" />

          {/* Eyebrows */}
          <path
            d={state === 'thinking' ? 'M 66 93 Q 78 86 88 92' : 'M 66 95 Q 78 90 88 94'}
            stroke="#1f1c18"
            strokeWidth="3.2"
            fill="none"
            strokeLinecap="round"
          />
          <path
            d={state === 'thinking' ? 'M 112 92 Q 122 86 134 93' : 'M 112 94 Q 122 90 134 95'}
            stroke="#1f1c18"
            strokeWidth="3.2"
            fill="none"
            strokeLinecap="round"
          />

          {/* Eyes */}
          {blink ? (
            <>
              <line x1="68" y1="106" x2="86" y2="106" stroke="#1f1c18" strokeWidth="3" strokeLinecap="round" />
              <line x1="114" y1="106" x2="132" y2="106" stroke="#1f1c18" strokeWidth="3" strokeLinecap="round" />
            </>
          ) : (
            <>
              <ellipse cx="77" cy="106" rx="9" ry="6.5" fill="#ffffff" />
              <circle cx={state === 'thinking' ? 79 : 77} cy={state === 'thinking' ? 104 : 106} r="4.2" fill="#2b1e16" />
              <circle cx="75.5" cy="104.5" r="1.4" fill="#ffffff" />

              <ellipse cx="123" cy="106" rx="9" ry="6.5" fill="#ffffff" />
              <circle cx={state === 'thinking' ? 125 : 123} cy={state === 'thinking' ? 104 : 106} r="4.2" fill="#2b1e16" />
              <circle cx="121.5" cy="104.5" r="1.4" fill="#ffffff" />
            </>
          )}

          {/* Nose */}
          <path d="M 100 106 L 97 123 Q 100 126 105 123" stroke="#9e6234" strokeWidth="2.4" fill="none" strokeLinecap="round" />

          {/* Friendly Mustache */}
          <path d="M 82 132 Q 100 126 118 132" stroke="#1f1c18" strokeWidth="3.5" fill="none" strokeLinecap="round" />

          {/* Dynamic Lip-Sync Mouth */}
          {renderMouth()}
        </g>
      </svg>

      <div className="kiosk-state-badge">
        <span className={`kiosk-dot ${state}`} />
        <strong>
          {state === 'idle' && 'Ready · Tap Mic or Type Below'}
          {state === 'listening' && 'Listening to you...'}
          {state === 'thinking' && 'Consulting Cultural Knowledge...'}
          {state === 'speaking' && 'Speaking Cultural Guidance...'}
        </strong>
      </div>
    </div>
  )
}
