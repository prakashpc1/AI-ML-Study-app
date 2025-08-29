
import React, { createContext, useContext, useState, useEffect, ReactNode } from 'react';
// FIX: Remove imports from firebaseService as the service no longer exports them, and the Firebase integration is deprecated.
// import { getFirebaseAuth, isFirebaseInitialized, onAuthStateChanged } from '../services/firebaseService';
import { FirebaseUser } from '../types';

interface AuthContextType {
    user: FirebaseUser | null;
    loading: boolean;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export const AuthProvider: React.FC<{ children: ReactNode }> = ({ children }) => {
    // FIX: Firebase auth is no longer used, so the user is always null.
    const user: FirebaseUser | null = null;
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        // This hook now simply returns a default non-authenticated state. We just stop the loading indicator.
        setLoading(false);
    }, []);

    return (
        <AuthContext.Provider value={{ user, loading }}>
            {children}
        </AuthContext.Provider>
    );
};

export const useAuth = (): AuthContextType => {
    const context = useContext(AuthContext);
    if (context === undefined) {
        throw new Error('useAuth must be used within an AuthProvider');
    }
    return context;
};