
import { useState, useEffect, useRef } from 'react';

// FIX: Add minimal type definitions for the Web Speech API, which are not included in standard TypeScript DOM libraries.
// This resolves errors about missing properties on `window` and undefined types.
interface SpeechRecognitionErrorEvent extends Event {
  error: string;
}

interface SpeechRecognitionEvent extends Event {
  resultIndex: number;
  results: {
    isFinal: boolean;
    [key: number]: {
      transcript: string;
    };
  }[];
}

interface SpeechRecognition extends EventTarget {
  continuous: boolean;
  interimResults: boolean;
  lang: string;
  start(): void;
  stop(): void;
  onresult: (event: SpeechRecognitionEvent) => void;
  onend: () => void;
  onerror: (event: SpeechRecognitionErrorEvent) => void;
}

declare var SpeechRecognition: {
  new(): SpeechRecognition;
};

declare global {
  interface Window {
    SpeechRecognition: typeof SpeechRecognition;
    webkitSpeechRecognition: typeof SpeechRecognition;
  }
}

interface SpeechRecognitionHook {
    transcript: string;
    listening: boolean;
    startListening: () => void;
    stopListening: () => void;
    browserSupportsSpeechRecognition: boolean;
}

const getSpeechRecognition = () => {
    if (typeof window !== 'undefined') {
        return window.SpeechRecognition || window.webkitSpeechRecognition;
    }
    return null;
};

export const useSpeechRecognition = (): SpeechRecognitionHook => {
    const [transcript, setTranscript] = useState('');
    const [listening, setListening] = useState(false);
    const recognitionRef = useRef<SpeechRecognition | null>(null);
    
    const SpeechRecognition = getSpeechRecognition();
    const browserSupportsSpeechRecognition = !!SpeechRecognition;

    useEffect(() => {
        if (!browserSupportsSpeechRecognition) {
            console.warn('Speech recognition not supported in this browser.');
            return;
        }

        const recognition = new SpeechRecognition();
        recognition.continuous = true;
        recognition.interimResults = true;
        recognition.lang = 'en-US';

        recognition.onresult = (event: SpeechRecognitionEvent) => {
            let finalTranscript = '';
            for (let i = event.resultIndex; i < event.results.length; ++i) {
                if (event.results[i].isFinal) {
                    finalTranscript += event.results[i][0].transcript;
                }
            }
            if (finalTranscript) {
                 setTranscript(prev => prev ? `${prev} ${finalTranscript}` : finalTranscript);
            }
        };
        
        recognition.onend = () => {
            setListening(false);
        };
        
        recognition.onerror = (event: SpeechRecognitionErrorEvent) => {
            console.error('Speech recognition error:', event.error);
            setListening(false);
        };

        recognitionRef.current = recognition;

        return () => {
            recognition.stop();
        };
    }, [browserSupportsSpeechRecognition]);

    const startListening = () => {
        if (recognitionRef.current && !listening) {
            setTranscript('');
            recognitionRef.current.start();
            setListening(true);
        }
    };

    const stopListening = () => {
        if (recognitionRef.current && listening) {
            recognitionRef.current.stop();
            setListening(false);
        }
    };

    return { transcript, listening, startListening, stopListening, browserSupportsSpeechRecognition };
};