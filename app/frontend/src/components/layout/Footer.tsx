import React from 'react';
import { Link } from 'react-router-dom';

interface FooterProps {
  asOfDate?: string;
}

export const Footer: React.FC<FooterProps> = ({ asOfDate = '2025-12-31' }) => {
  return (
    <footer className="w-full border-t bg-card text-muted-foreground py-8 px-4 sm:px-6 lg:px-8 mt-auto text-xs">
      <div className="max-w-7xl mx-auto flex flex-col md:flex-row items-center justify-between gap-4 text-center md:text-left">
        <div className="space-y-1.5 max-w-2xl">
          <p className="font-semibold text-foreground">
            Stock Pulse &copy; {new Date().getFullYear()} — Colombo Stock Exchange Upside Analytics
          </p>
          <p className="text-[11px] leading-relaxed text-muted-foreground">
            Stock Pulse provides data-driven rankings for information only. It is not financial advice. Signals shown have limited historical reliability.
          </p>
        </div>

        <div className="flex flex-col sm:flex-row items-center gap-4 text-[11px]">
          <span className="tabular-nums">Market data as of: <strong className="text-foreground">{asOfDate}</strong></span>
          <div className="flex items-center gap-3">
            <Link to="/how-it-works" className="hover:text-primary underline-offset-4 hover:underline">
              Methodology
            </Link>
            <span>&bull;</span>
            <Link to="/public/info" className="hover:text-primary underline-offset-4 hover:underline">
              System Info
            </Link>
          </div>
        </div>
      </div>
    </footer>
  );
};
