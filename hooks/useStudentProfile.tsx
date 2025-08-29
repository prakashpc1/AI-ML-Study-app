import React, { createContext, useContext, useState, useEffect, ReactNode, useCallback } from 'react';
import { StudentProfile } from '../types';
import { loadStudentProfile, saveStudentProfile, clearStudentProfile as clearProfileStorage } from '../services/profile';

interface StudentProfileContextType {
    studentProfile: StudentProfile | null;
    loading: boolean;
    setStudentProfile: (profile: StudentProfile) => void;
    clearStudentProfile: () => void;
}

const StudentProfileContext = createContext<StudentProfileContextType | undefined>(undefined);

export const StudentProfileProvider: React.FC<{ children: ReactNode }> = ({ children }) => {
    const [studentProfile, setProfile] = useState<StudentProfile | null>(null);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        // On initial load, try to get the profile from localStorage
        const profile = loadStudentProfile();
        if (profile) {
            setProfile(profile);
        }
        setLoading(false);
    }, []);

    const setStudentProfile = useCallback((profile: StudentProfile) => {
        saveStudentProfile(profile);
        setProfile(profile);
    }, []);

    const clearStudentProfile = useCallback(() => {
        clearProfileStorage();
        setProfile(null);
    }, []);

    return (
        <StudentProfileContext.Provider value={{ studentProfile, loading, setStudentProfile, clearStudentProfile }}>
            {children}
        </StudentProfileContext.Provider>
    );
};

export const useStudentProfile = (): StudentProfileContextType => {
    const context = useContext(StudentProfileContext);
    if (context === undefined) {
        throw new Error('useStudentProfile must be used within a StudentProfileProvider');
    }
    return context;
};
