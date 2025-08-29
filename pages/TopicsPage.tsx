
import React, { useState, useMemo } from 'react';
import { TOPIC_CATEGORIES } from '../constants';
import SearchBar from '../components/SearchBar';
import TopicCard from '../components/TopicCard';
import { TopicCategory } from '../types';
import PageHeader from '../components/PageHeader';

const TopicsPage: React.FC = () => {
    const [searchTerm, setSearchTerm] = useState('');

    const filteredCategories = useMemo(() => {
        if (!searchTerm) {
            return TOPIC_CATEGORIES;
        }

        const lowercasedFilter = searchTerm.toLowerCase();
        const result: TopicCategory[] = [];

        TOPIC_CATEGORIES.forEach(category => {
            const filteredTopics = category.topics.filter(topic =>
                topic.title.toLowerCase().includes(lowercasedFilter) ||
                topic.description.toLowerCase().includes(lowercasedFilter)
            );

            if (filteredTopics.length > 0) {
                result.push({ ...category, topics: filteredTopics });
            }
        });

        return result;
    }, [searchTerm]);

    return (
        <div className="p-4 sm:p-6 lg:p-12 animate-fade-in">
            <div className="max-w-7xl mx-auto">
                <PageHeader title="All Topics" />
                <div className="mb-8 max-w-lg">
                    <SearchBar searchTerm={searchTerm} onSearchChange={setSearchTerm} />
                </div>

                {filteredCategories.length > 0 ? (
                    <div className="space-y-12">
                        {filteredCategories.map(category => (
                            <section key={category.name}>
                                <h2 className="text-2xl font-bold mb-6 pb-2 border-b-2 border-gray-800 text-white">
                                    {category.name}
                                </h2>
                                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
                                    {category.topics.map(topic => (
                                        <TopicCard key={topic.id} topic={topic} />
                                    ))}
                                </div>
                            </section>
                        ))}
                    </div>
                ) : (
                    <div className="text-center py-20 bg-gray-900 rounded-xl">
                        <p className="text-lg text-gray-500">No topics found for "{searchTerm}".</p>
                    </div>
                )}
            </div>
        </div>
    );
};

export default TopicsPage;