import React, { useState, useEffect } from 'react';
import { StudentProfile } from '../types';
import { AcademyLogoIcon } from './icons/Icons';
import LoadingSpinner from './LoadingSpinner';

interface StudentGateProps {
    onProfileReady: (profile: StudentProfile) => void;
    existingProfile?: StudentProfile | null;
    onClearProfile?: () => void;
}

const semesterOptions = ["1st", "2nd", "3rd", "4th", "5th", "6th", "7th", "8th"];
const languageOptions = ["English", "Hindi", "Kannada", "Telugu", "Tamil", "Marathi", "Bengali"];

const StudentGate: React.FC<StudentGateProps> = ({ onProfileReady, existingProfile, onClearProfile }) => {
    const [profile, setProfile] = useState<Partial<StudentProfile>>(
        existingProfile || {
            program: 'CSE (AIML)',
            preferredLanguage: 'English',
            consent: false,
        }
    );
    const [errors, setErrors] = useState<Partial<Record<keyof StudentProfile, string>>>({});
    const [isSubmitting, setIsSubmitting] = useState(false);

    useEffect(() => {
        if (existingProfile) {
            setProfile(existingProfile);
        }
    }, [existingProfile]);
    
    const validate = (): boolean => {
        const newErrors: Partial<Record<keyof StudentProfile, string>> = {};
        
        if (!profile.fullName || profile.fullName.length < 2 || profile.fullName.length > 60) {
            newErrors.fullName = "Full Name must be between 2 and 60 characters.";
        }
        
        const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
        if (profile.email && !emailRegex.test(profile.email)) {
            newErrors.email = "Please enter a valid email address format.";
        }
        
        if (!profile.phone) {
            newErrors.phone = "Phone number is required.";
        } else if (!/^\d{10,15}$/.test(profile.phone)) {
            newErrors.phone = "Phone number must be 10-15 digits.";
        }
        
        if (!profile.rollNumber) {
            newErrors.rollNumber = "Roll Number is required.";
        }

        if (!profile.institution) newErrors.institution = "Institution is required.";
        if (!profile.program) newErrors.program = "Program is required.";
        if (!profile.semester) newErrors.semester = "Semester is required.";
        if (!profile.consent) newErrors.consent = "You must agree to continue.";

        setErrors(newErrors);
        return Object.keys(newErrors).length === 0;
    };


    const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement | HTMLTextAreaElement>) => {
        const { name, value, type } = e.target;
        const isCheckbox = type === 'checkbox';
        setProfile(prev => ({
            ...prev,
            [name]: isCheckbox ? (e.target as HTMLInputElement).checked : value,
        }));
    };
    
    const handleSubmit = (e: React.FormEvent) => {
        e.preventDefault();
        if (validate()) {
            setIsSubmitting(true);
            const finalProfile: StudentProfile = {
                ...profile,
                email: profile.email?.toLowerCase(),
            } as StudentProfile;
            
            // Simulate a quick save operation
            setTimeout(() => {
                onProfileReady(finalProfile);
                setIsSubmitting(false);
            }, 500);
        }
    };

    return (
        <div className="fixed inset-0 z-50 bg-black flex items-center justify-center p-4">
            <div className="w-full max-w-2xl bg-gray-900 border border-gray-800 rounded-2xl p-6 sm:p-8 max-h-[90vh] overflow-y-auto">
                 <div className="text-center mb-6">
                    <AcademyLogoIcon className="w-12 h-12 text-white mx-auto mb-3" />
                    <h1 className="text-2xl sm:text-3xl font-bold text-white">Welcome to AI and ML study</h1>
                    <p className="text-gray-400 mt-2">
                        {existingProfile ? "Update your profile details." : "Please fill out your details to get started."}
                    </p>
                </div>
                
                <form onSubmit={handleSubmit} className="space-y-4">
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                        <InputField label="Full Name" name="fullName" value={profile.fullName} onChange={handleChange} error={errors.fullName} required />
                        <InputField label="Email (Optional)" name="email" type="email" value={profile.email} onChange={handleChange} error={errors.email} />
                    </div>
                     <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                        <InputField label="Institution" name="institution" value={profile.institution} onChange={handleChange} error={errors.institution} required />
                        <InputField label="Roll Number" name="rollNumber" value={profile.rollNumber} onChange={handleChange} error={errors.rollNumber} required />
                    </div>
                     <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                        <InputField label="Program (e.g., CSE)" name="program" value={profile.program} onChange={handleChange} error={errors.program} required />
                        <SelectField label="Semester" name="semester" value={profile.semester} options={semesterOptions} onChange={handleChange} error={errors.semester} required />
                    </div>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                        <InputField label="Phone" name="phone" type="tel" value={profile.phone} onChange={handleChange} error={errors.phone} required />
                         <SelectField label="Preferred Language" name="preferredLanguage" value={profile.preferredLanguage} options={languageOptions} onChange={handleChange} />
                    </div>
                    <div>
                        <label className="block text-sm font-medium text-gray-300 mb-1">Study Goals (Optional)</label>
                        <textarea
                            name="goals"
                            value={profile.goals || ''}
                            onChange={handleChange}
                            rows={3}
                            className="w-full bg-gray-800 border border-gray-700 rounded-lg py-2 px-3 text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
                        />
                    </div>

                    <div className="pt-2">
                        <label className="flex items-center">
                            <input type="checkbox" name="consent" checked={!!profile.consent} onChange={handleChange} className="h-4 w-4 rounded bg-gray-700 border-gray-600 text-blue-500 focus:ring-blue-500" />
                            <span className="ml-3 text-sm text-gray-300">I agree to use my information to personalize my study experience.*</span>
                        </label>
                        {errors.consent && <p className="text-xs text-red-400 mt-1">{errors.consent}</p>}
                    </div>
                    
                    <div className="pt-4 flex flex-col sm:flex-row items-center gap-4">
                         <button type="submit" disabled={isSubmitting} className="w-full sm:w-auto bg-blue-600 hover:bg-blue-700 text-white font-bold py-3 px-8 rounded-lg transition-colors disabled:bg-gray-500 flex items-center justify-center">
                            {isSubmitting ? <LoadingSpinner /> : (existingProfile ? "Update Profile" : "Save & Continue")}
                        </button>
                         {onClearProfile && (
                            <button type="button" onClick={onClearProfile} className="text-sm text-red-500 hover:underline">
                                Clear Profile & Reset
                            </button>
                        )}
                    </div>
                </form>
                 <p className="text-xs text-gray-500 text-center mt-6">
                    Data is stored only on your device (localStorage). You can clear it anytime.
                </p>
            </div>
        </div>
    );
};

// Helper components for form fields
const InputField = ({ label, name, type = 'text', value, onChange, error, required = false }: any) => (
    <div>
        <label htmlFor={name} className="block text-sm font-medium text-gray-300 mb-1">
            {label}{required && '*'}
        </label>
        <input
            id={name}
            name={name}
            type={type}
            value={value || ''}
            onChange={onChange}
            className={`w-full bg-gray-800 border ${error ? 'border-red-500' : 'border-gray-700'} rounded-lg py-2 px-3 text-white focus:outline-none focus:ring-2 focus:ring-blue-500`}
        />
        {error && <p className="text-xs text-red-400 mt-1">{error}</p>}
    </div>
);

const SelectField = ({ label, name, value, options, onChange, error, required = false }: any) => (
     <div>
        <label htmlFor={name} className="block text-sm font-medium text-gray-300 mb-1">
            {label}{required && '*'}
        </label>
        <select
            id={name}
            name={name}
            value={value || ''}
            onChange={onChange}
            className={`w-full bg-gray-800 border ${error ? 'border-red-500' : 'border-gray-700'} rounded-lg py-2 px-3 text-white focus:outline-none focus:ring-2 focus:ring-blue-500`}
        >
            <option value="" disabled>Select...</option>
            {options.map((opt: string) => <option key={opt} value={opt}>{opt}</option>)}
        </select>
        {error && <p className="text-xs text-red-400 mt-1">{error}</p>}
    </div>
);


export default StudentGate;