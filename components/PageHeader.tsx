
import React from 'react';
import { useNavigate } from 'react-router-dom';
import { ChevronLeft } from './icons/Icons';

interface PageHeaderProps {
    title: string;
}

const PageHeader: React.FC<PageHeaderProps> = ({ title }) => {
    const navigate = useNavigate();

    return (
        <div className="relative flex items-center justify-center mb-8">
            <button 
                onClick={() => navigate(-1)} 
                className="absolute left-0 p-2 rounded-full hover:bg-gray-800 transition-colors"
                aria-label="Go back"
            >
                <ChevronLeft className="h-6 w-6 text-gray-300" />
            </button>
            <h1 className="text-4xl font-extrabold text-white text-center">{title}</h1>
        </div>
    );
};

export default PageHeader;