import React from 'react';

export interface StudentProfile {
  fullName: string;
  email?: string;
  phone: string;
  institution: string;
  program: string;
  semester: string;
  rollNumber: string;
  preferredLanguage?: string;
  goals?: string;
  consent: boolean;
}

export interface Topic {
    id: string;
    title: string;
    category: string;
    description: string;
    content: string; // Could be markdown or plain text
    icon?: React.FC<React.SVGProps<SVGSVGElement>>;
    color?: string; // e.g. 'text-orange-500'
    chartData?: { name: string; value: number }[];
    bookmarked?: boolean;
}

export interface TopicCategory {
    name: string;
    topics: Topic[];
}

export interface ChatMessage {
    id:string;
    text: string;
    sender: 'user' | 'ai';
    timestamp: number;
}

export interface QuizQuestion {
    question: string;
    options: string[];
    answer: number; // index of the correct option
}

// FIX: Add FirebaseUser type definition to resolve an import error. This is likely a remnant of a removed Firebase integration.
export interface FirebaseUser {
    uid: string;
    email: string | null;
    displayName: string | null;
}


export enum Theme {
    Light = 'light',
    Dark = 'dark'
}