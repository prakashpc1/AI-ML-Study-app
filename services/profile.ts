
import { StudentProfile } from '../types';

const PROFILE_STORAGE_KEY = 'student_profile_v1';

/**
 * Validates that the loaded object conforms to the StudentProfile interface.
 * This is a basic check to guard against malformed data in localStorage.
 * @param obj The object to validate.
 * @returns True if the object is a valid StudentProfile, false otherwise.
 */
const isValidProfile = (obj: any): obj is StudentProfile => {
    return (
        typeof obj === 'object' &&
        obj !== null &&
        typeof obj.fullName === 'string' &&
        (obj.email === undefined || typeof obj.email === 'string') &&
        typeof obj.phone === 'string' &&
        typeof obj.institution === 'string' &&
        typeof obj.program === 'string' &&
        typeof obj.semester === 'string' &&
        typeof obj.rollNumber === 'string' &&
        typeof obj.consent === 'boolean'
    );
};

/**
 * Saves the student's profile to localStorage.
 * @param profile The student profile data to save.
 */
export const saveStudentProfile = (profile: StudentProfile): void => {
    try {
        const profileJson = JSON.stringify(profile);
        localStorage.setItem(PROFILE_STORAGE_KEY, profileJson);
    } catch (error) {
        console.error("Failed to save student profile to localStorage:", error);
    }
};

/**
 * Loads the student's profile from localStorage.
 * @returns The student profile data if it exists and is valid, otherwise null.
 */
export const loadStudentProfile = (): StudentProfile | null => {
    try {
        const profileJson = localStorage.getItem(PROFILE_STORAGE_KEY);
        if (!profileJson) {
            return null;
        }
        const profile = JSON.parse(profileJson);
        
        if (isValidProfile(profile)) {
            return profile;
        } else {
            console.warn("Invalid student profile found in localStorage. Clearing it.");
            clearStudentProfile();
            return null;
        }
    } catch (error) {
        console.error("Failed to load student profile from localStorage:", error);
        return null;
    }
};

/**
 * Removes the student's profile from localStorage.
 */
export const clearStudentProfile = (): void => {
    try {
        localStorage.removeItem(PROFILE_STORAGE_KEY);
    } catch (error) {
        console.error("Failed to clear student profile from localStorage:", error);
    }
};