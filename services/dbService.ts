import React from 'react';
import { openDB, DBSchema, IDBPDatabase } from 'idb';
import { Topic, ChatMessage } from '../types';
import { TOPIC_CATEGORIES } from '../constants';

const DB_NAME = 'AIStudyHubDB';
const DB_VERSION = 1;
const TOPICS_STORE = 'topics';
const BOOKMARKS_STORE = 'bookmarks';
const CHAT_HISTORY_STORE = 'chatHistory';

// Create a flat map for easy lookup during rehydration
const allTopicsWithIcons = TOPIC_CATEGORIES.flatMap(cat => cat.topics);
// FIX: Add 'React' import to resolve a complex type error. The error indicates two different definitions of React types are being used, which can happen when React is not explicitly imported in a .ts file. This should unify the type definitions and allow the Map constructor to be called correctly.
const topicIconMap = new Map<string, React.FC<React.SVGProps<SVGSVGElement>> | undefined>(
    allTopicsWithIcons.map(t => [t.id, t.icon])
);

// Helper function to strip icon before saving to DB
const dehydrateTopic = (topic: Topic): Omit<Topic, 'icon'> => {
    const { icon, ...rest } = topic;
    return rest;
};

// Helper function to add icon back after fetching from DB
// A simple type assertion is used here for convenience.
const rehydrateTopic = (topic: Omit<Topic, 'icon'> | undefined | null): Topic | undefined => {
    if (!topic) return undefined;
    const icon = topicIconMap.get(topic.id);
    return { ...topic, icon } as Topic;
};


interface MyDB extends DBSchema {
    [TOPICS_STORE]: {
        key: string;
        value: Omit<Topic, 'icon'>; // Store dehydrated topics
    };
    [BOOKMARKS_STORE]: {
        key: string;
        value: Omit<Topic, 'icon'>; // Store dehydrated topics
    };
    [CHAT_HISTORY_STORE]: {
        key: string;
        value: ChatMessage;
        indexes: { 'by-timestamp': number };
    };
}

let dbPromise: Promise<IDBPDatabase<MyDB>>;

const getDb = (): Promise<IDBPDatabase<MyDB>> => {
    if (!dbPromise) {
        dbPromise = openDB<MyDB>(DB_NAME, DB_VERSION, {
            upgrade(db) {
                if (!db.objectStoreNames.contains(TOPICS_STORE)) {
                    db.createObjectStore(TOPICS_STORE, { keyPath: 'id' });
                }
                if (!db.objectStoreNames.contains(BOOKMARKS_STORE)) {
                    db.createObjectStore(BOOKMARKS_STORE, { keyPath: 'id' });
                }
                if (!db.objectStoreNames.contains(CHAT_HISTORY_STORE)) {
                    const chatStore = db.createObjectStore(CHAT_HISTORY_STORE, { keyPath: 'id' });
                    chatStore.createIndex('by-timestamp', 'timestamp');
                }
            },
        });
    }
    return dbPromise;
};

// Populate DB with initial topics if empty
const populateInitialData = async () => {
    const db = await getDb();
    const count = await db.count(TOPICS_STORE);
    if (count === 0) {
        const tx = db.transaction(TOPICS_STORE, 'readwrite');
        const allTopics = TOPIC_CATEGORIES.flatMap(cat => cat.topics);
        await Promise.all(allTopics.map(topic => tx.store.put(dehydrateTopic(topic))));
        await tx.done;
    }
};

populateInitialData();


// --- Topic Functions ---
export const getTopicById = async (id: string): Promise<Topic | undefined> => {
    const db = await getDb();
    const dehydratedTopic = await db.get(TOPICS_STORE, id);
    return rehydrateTopic(dehydratedTopic);
};

// --- Bookmark Functions ---
export const toggleBookmark = async (topic: Topic): Promise<boolean> => {
    const db = await getDb();
    const existing = await db.get(BOOKMARKS_STORE, topic.id);
    if (existing) {
        await db.delete(BOOKMARKS_STORE, topic.id);
        return false;
    } else {
        await db.put(BOOKMARKS_STORE, dehydrateTopic(topic));
        return true;
    }
};

export const getBookmarkStatus = async (topicId: string): Promise<boolean> => {
    const db = await getDb();
    const bookmark = await db.get(BOOKMARKS_STORE, topicId);
    return !!bookmark;
};

export const getBookmarks = async (): Promise<Topic[]> => {
    const db = await getDb();
    const dehydratedBookmarks = await db.getAll(BOOKMARKS_STORE);
    return dehydratedBookmarks.map(t => rehydrateTopic(t)).filter(Boolean) as Topic[];
};

// --- Chat History Functions ---
export const saveChatMessage = async (message: ChatMessage): Promise<void> => {
    const db = await getDb();
    await db.put(CHAT_HISTORY_STORE, message);
};

export const getChatHistory = async (): Promise<ChatMessage[]> => {
    const db = await getDb();
    return db.getAllFromIndex(CHAT_HISTORY_STORE, 'by-timestamp');
};

/*
    NOTE on Firebase Sync:
    To sync with Firebase, you would create functions here that:
    1. On app load, fetch data from Firestore and update IndexedDB.
    2. When offline, all writes go to IndexedDB.
    3. When online again (using browser's online/offline events), read any "dirty" records from IndexedDB 
       and push them to Firestore.
    4. Listen for real-time updates from Firestore (onSnapshot) and update IndexedDB accordingly.
    This provides a seamless online/offline experience.
*/