
import React from 'react';
import { useStudentProfile } from '../hooks/useStudentProfile';
import StudentGate from '../components/StudentGate';
import PageHeader from '../components/PageHeader';

const SettingsPage: React.FC = () => {
    const { studentProfile, setStudentProfile, clearStudentProfile } = useStudentProfile();

    if (!studentProfile) {
        // This case should ideally not be reached if AppInitializer works correctly,
        // but it's a good fallback.
        return <StudentGate onProfileReady={setStudentProfile} />;
    }

    return (
        <div className="p-4 sm:p-6 lg:p-8 animate-fade-in">
             <div className="max-w-4xl mx-auto">
                <PageHeader title="Your Profile" />
                <p className="text-center text-gray-400 mb-8">
                    This information is stored only on your device and helps personalize your learning experience.
                </p>
                <StudentGate
                    existingProfile={studentProfile}
                    onProfileReady={setStudentProfile}
                    onClearProfile={clearStudentProfile}
                />
            </div>
        </div>
    );
};

export default SettingsPage;