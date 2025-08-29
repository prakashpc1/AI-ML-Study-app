
import React from 'react';
import { useNavigate } from 'react-router-dom';
import { Topic } from '../types';

interface TopicCardProps {
    topic: Topic;
}

const gradientMap: { [key: string]: string } = {
    'text-orange-500': 'from-orange-400 to-red-500',
    'text-teal-500': 'from-teal-400 to-cyan-500',
    'text-pink-500': 'from-pink-400 to-rose-500',
    'text-green-500': 'from-green-400 to-emerald-500',
    'text-purple-500': 'from-purple-400 to-indigo-500',
    'text-sky-400': 'from-sky-400 to-blue-500',
};

const TopicCard: React.FC<TopicCardProps> = ({ topic }) => {
    const navigate = useNavigate();
    const gradient = gradientMap[topic.color || 'text-purple-500'] || 'from-gray-400 to-gray-500';
    const lessons = topic.id === 'drought-prediction-lstm' ? 7 : 5; // Example lesson count

    return (
        <div
            onClick={() => navigate(`/topics/${topic.id}`)}
            className={`relative p-6 rounded-2xl overflow-hidden group transition-all duration-300 ease-in-out transform hover:scale-105 hover:shadow-2xl cursor-pointer bg-gradient-to-br ${gradient}`}
        >
            <div 
                className="absolute inset-0 bg-repeat bg-center opacity-5"
                style={{ backgroundImage: `url("data:image/svg+xml,%3Csvg width='60' height='60' viewBox='0 0 60 60' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='none' fill-rule='evenodd'%3E%3Cg fill='%23ffffff' fill-opacity='1'%3E%3Cpath d='M36 34v-4h-2v4h-4v2h4v4h2v-4h4v-2h-4zm0-30V0h-2v4h-4v2h4v4h2V6h4V4h-4zM6 34v-4H4v4H0v2h4v4h2v-4h4v-2H6zM6 4V0H4v4H0v2h4v4h2V6h4V4H6z'/%3E%3C/g%3E%3C/g%3E%3C/svg%3E")`}}
            ></div>
            
            <div className="relative z-10">
                <div className="flex justify-between items-start mb-4">
                    <div className="w-14 h-14 bg-white/20 backdrop-blur-md rounded-xl flex items-center justify-center">
                        {topic.icon && <topic.icon className="h-8 w-8 text-white" />}
                    </div>
                </div>

                <h3 className="text-xl font-bold text-white mb-2">{topic.title}</h3>
                <p className="text-sm text-white/80 h-10">{topic.description}</p>
                
                <div className="mt-6 flex items-center justify-between">
                     <div className="flex items-center">
                        <div className="flex -space-x-1 overflow-hidden">
                           {[...Array(lessons)].map((_, i) => (
                             <div key={i} className={`w-2 h-2 rounded-full ${i < 5 ? 'bg-white' : 'bg-white/50'}`}></div>
                           ))}
                        </div>
                         <span className="text-xs text-white/80 ml-2">{lessons} lessons</span>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default TopicCard;