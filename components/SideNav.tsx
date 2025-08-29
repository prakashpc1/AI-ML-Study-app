import React from 'react';
import { NavLink } from 'react-router-dom';
import { useStudentProfile } from '../hooks/useStudentProfile';
import { AcademyLogoIcon, Dashboard, Book, Bot, Bookmark, Settings, LogOut, User } from './icons/Icons';

const SideNav: React.FC = () => {
    const { studentProfile, clearStudentProfile } = useStudentProfile();

    const navItems = [
        { path: '/', label: 'Dashboard', icon: Dashboard },
        { path: '/topics', label: 'Topics', icon: Book },
        { path: '/ai-assistant', label: 'AI Assistant', icon: Bot },
        { path: '/bookmarks', label: 'Bookmarks', icon: Bookmark },
    ];

    const commonClass = "flex items-center w-full h-12 px-4 text-sm font-medium rounded-lg transition-colors duration-200";
    const activeClass = "bg-gray-800 text-white";
    const inactiveClass = "text-gray-400 hover:bg-gray-800 hover:text-white";

    return (
        <aside className="fixed top-0 left-0 z-50 w-64 h-full bg-gray-900 border-r border-gray-800 flex flex-col p-4">
            <div className="flex items-center space-x-3 h-16 px-4 mb-8">
                <AcademyLogoIcon className="h-8 w-8 text-white" />
                <div>
                    <h1 className="text-lg font-bold text-white">AI and ML study</h1>
                    <p className="text-xs text-gray-400">Explore the World of AI</p>
                </div>
            </div>

            <nav className="flex-grow space-y-2">
                {navItems.map(item => (
                    <NavLink
                        key={item.path}
                        to={item.path}
                        end={item.path === '/'}
                        className={({ isActive }) => `${commonClass} ${isActive ? activeClass : inactiveClass}`}
                    >
                        <item.icon className="w-5 h-5 mr-3" />
                        <span>{item.label}</span>
                    </NavLink>
                ))}
            </nav>

            <div className="mt-auto">
                 <NavLink
                    to="/settings"
                    className={({ isActive }) => `${commonClass} ${isActive ? activeClass : inactiveClass}`}
                >
                    <Settings className="w-5 h-5 mr-3" />
                    <span>Edit Profile</span>
                </NavLink>
                {studentProfile && (
                    <div className="border-t border-gray-800 mt-4 pt-4 flex items-center justify-between">
                         <div className="flex items-center gap-3 overflow-hidden">
                            <div className="w-8 h-8 bg-gray-700 rounded-full flex items-center justify-center flex-shrink-0">
                                <User className="w-4 h-4 text-gray-400" />
                            </div>
                             <div className="overflow-hidden">
                                <p className="text-sm font-semibold text-gray-200 truncate">{studentProfile.fullName}</p>
                                <p className="text-xs text-gray-400 truncate">{studentProfile.email || studentProfile.phone}</p>
                             </div>
                        </div>
                        <button onClick={clearStudentProfile} className="text-gray-400 hover:text-white p-2 rounded-lg hover:bg-gray-800 flex-shrink-0" title="Clear Profile & Reset">
                           <LogOut className="w-5 h-5" />
                        </button>
                    </div>
                )}
            </div>
        </aside>
    );
};

export default SideNav;