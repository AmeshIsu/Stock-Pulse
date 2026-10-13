import React, { useEffect, useState } from 'react';
import { Outlet } from 'react-router-dom';
import { Navbar } from './Navbar';
import { Footer } from './Footer';
import { FreshnessBanner } from './FreshnessBanner';
import { DisclaimerModal } from './DisclaimerModal';
import { apiClient } from '../../lib/api';
import { PublicInfo } from '../../lib/types';

export const AppLayout: React.FC = () => {
  const [asOfDate, setAsOfDate] = useState<string>('2025-12-31');

  useEffect(() => {
    // Fetch public metadata to retrieve canonical as-of date
    apiClient<PublicInfo>('/public/info')
      .then((data) => {
        if (data.data_as_of) {
          setAsOfDate(data.data_as_of);
        }
      })
      .catch(() => {
        // Fallback default
      });
  }, []);

  return (
    <div className="min-h-screen flex flex-col bg-background text-foreground">
      <FreshnessBanner asOfDate={asOfDate} />
      <Navbar />
      <main className="flex-1 w-full max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
        <Outlet />
      </main>
      <Footer asOfDate={asOfDate} />
      <DisclaimerModal />
    </div>
  );
};
