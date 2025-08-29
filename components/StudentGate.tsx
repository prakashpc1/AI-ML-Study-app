
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
            
            setTimeout(() => {
                onProfileReady(finalProfile);
                setIsSubmitting(false);
            }, 500);
        }
    };

    return (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-gray-900 overflow-hidden">
             <div className="absolute inset-0 bg-gradient-to-br from-indigo-500/30 via-purple-500/30 to-pink-500/30 animate-gradient-xy"></div>
            <div className="w-full max-w-md bg-black/40 backdrop-blur-xl border border-white/20 rounded-3xl p-6 sm:p-10 shadow-2xl max-h-[90vh] overflow-y-auto">
                 <div className="text-center mb-8">
                    <AcademyLogoIcon className="w-16 h-16 text-white mx-auto mb-4" />
                    <h1 className="text-4xl sm:text-5xl font-extrabold text-white">
                        {existingProfile ? "Your Profile" : "Get Started"}
                    </h1>
                    <p className="text-gray-300 mt-4 text-lg">
                        {existingProfile ? "Update your details below." : "Create your local profile to begin."}
                    </p>
                </div>
                
                <form onSubmit={handleSubmit} className="space-y-5">
                    <InputField label="Full Name" name="fullName" value={profile.fullName} onChange={handleChange} error={errors.fullName} required />
                    <InputField label="Institution" name="institution" value={profile.institution} onChange={handleChange} error={errors.institution} required />
                    <div className="grid grid-cols-1 sm:grid-cols-2 gap-5">
                        <InputField label="Roll Number" name="rollNumber" value={profile.rollNumber} onChange={handleChange} error={errors.rollNumber} required />
                        <InputField label="Phone" name="phone" type="tel" value={profile.phone} onChange={handleChange} error={errors.phone} required />
                    </div>
                    <div className="grid grid-cols-1 sm:grid-cols-2 gap-5">
                        <InputField label="Program (e.g., CSE)" name="program" value={profile.program} onChange={handleChange} error={errors.program} required />
                        <SelectField label="Semester" name="semester" value={profile.semester} options={semesterOptions} onChange={handleChange} error={errors.semester} required />
                    </div>
                     <div className="grid grid-cols-1 sm:grid-cols-2 gap-5">
                        <InputField label="Email (Optional)" name="email" type="email" value={profile.email} onChange={handleChange} error={errors.email} />
                        <SelectField label="Preferred Language" name="preferredLanguage" value={profile.preferredLanguage} options={languageOptions} onChange={handleChange} />
                    </div>

                    <div className="pt-2">
                        <label className="flex items-center">
                            <input type="checkbox" name="consent" checked={!!profile.consent} onChange={handleChange} className="h-4 w-4 rounded bg-gray-700 border-gray-600 text-purple-500 focus:ring-purple-500" />
                            <span className="ml-3 text-sm text-gray-300">I agree to use my info to personalize my study experience.*</span>
                        </label>
                        {errors.consent && <p className="text-xs text-red-400 mt-1">{errors.consent}</p>}
                    </div>
                    
                    <div className="pt-6 flex flex-col sm:flex-row items-center gap-4">
                         <button type="submit" disabled={isSubmitting} className="w-full sm:w-auto bg-gradient-to-r from-purple-600 to-blue-600 hover:from-purple-700 hover:to-blue-700 text-white font-bold py-3 px-8 rounded-full transition-all duration-300 transform hover:scale-105 shadow-lg disabled:bg-gray-500 flex items-center justify-center">
                            {isSubmitting ? <LoadingSpinner /> : (existingProfile ? "Update Profile" : "Save & Continue")}
                        </button>
                         {onClearProfile && (
                            <button type="button" onClick={onClearProfile} className="text-sm text-red-500 hover:underline">
                                Clear Profile & Reset
                            </button>
                        )}
                    </div>
                </form>
                 <p className="text-xs text-gray-500 text-center mt-8">
                    Data is stored only on your device (localStorage).
                </p>
            </div>
        </div>
    );
};

// Helper components for form fields
const InputField = ({ label, name, type = 'text', value, onChange, error, required = false }: any) => (
    <div>
        <label htmlFor={name} className="block text-sm font-semibold text-gray-300 mb-2">
            {label}{required && <span className="text-red-400">*</span>}
        </label>
        <input
            id={name}
            name={name}
            type={type}
            value={value || ''}
            onChange={onChange}
            className={`w-full bg-white/5 border-2 ${error ? 'border-red-500/50' : 'border-white/20'} rounded-lg py-2.5 px-4 text-white focus:outline-none focus:ring-2 focus:ring-purple-500 backdrop-blur-sm transition-colors duration-300`}
        />
        {error && <p className="text-xs text-red-400 mt-1">{error}</p>}
    </div>
);

const SelectField = ({ label, name, value, options, onChange, error, required = false }: any) => (
     <div>
        <label htmlFor={name} className="block text-sm font-semibold text-gray-300 mb-2">
            {label}{required && <span className="text-red-400">*</span>}
        </label>
        <select
            id={name}
            name={name}
            value={value || ''}
            onChange={onChange}
            className={`w-full bg-white/5 border-2 ${error ? 'border-red-500/50' : 'border-white/20'} rounded-lg py-2.5 px-4 text-white focus:outline-none focus:ring-2 focus:ring-purple-500 backdrop-blur-sm transition-colors duration-300`}
        >
            <option value="" disabled>Select...</option>
            {options.map((opt: string) => <option key={opt} value={opt} className="bg-gray-800">{opt}</option>)}
        </select>
        {error && <p className="text-xs text-red-400 mt-1">{error}</p>}
    </div>
);


export default StudentGate;