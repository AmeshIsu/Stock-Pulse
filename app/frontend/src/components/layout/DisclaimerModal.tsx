import React, { useState } from 'react';
import { useAuth } from '../../contexts/AuthContext';
import { AlertTriangle, CheckCircle } from 'lucide-react';

export const DisclaimerModal: React.FC = () => {
  const { isAuthenticated, hasAcknowledgedDisclaimer, acknowledgeDisclaimer } = useAuth();
  const [agreed, setAgreed] = useState(false);

  if (!isAuthenticated || hasAcknowledgedDisclaimer) {
    return null;
  }

  const handleConfirm = () => {
    if (agreed) {
      acknowledgeDisclaimer();
    }
  };

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm p-4"
      role="dialog"
      aria-modal="true"
      aria-labelledby="disclaimer-modal-title"
      data-testid="disclaimer-modal"
    >
      <div className="bg-card text-card-foreground border border-border rounded-xl shadow-2xl max-w-lg w-full p-6 animate-in fade-in zoom-in-95 duration-200">
        <div className="flex items-center gap-3 text-amber-600 dark:text-amber-400 mb-4">
          <AlertTriangle className="w-6 h-6" aria-hidden="true" />
          <h2 id="disclaimer-modal-title" className="text-lg font-bold text-foreground">
            Important Information & Disclaimer
          </h2>
        </div>

        <p className="text-sm text-muted-foreground leading-relaxed mb-4">
          Welcome to Stock Pulse. Before viewing our data-driven rankings and Colombo Stock Exchange analytics, please review and acknowledge our informational notice:
        </p>

        <div className="p-4 bg-slate-50 dark:bg-slate-900 border border-border rounded-lg text-sm font-medium text-foreground mb-5">
          “Stock Pulse provides data-driven rankings for information only. It is not financial advice. Signals shown have limited historical reliability.”
        </div>

        <div className="space-y-4">
          <label className="flex items-start gap-3 cursor-pointer select-none">
            <input
              type="checkbox"
              id="disclaimer-checkbox"
              data-testid="disclaimer-checkbox"
              checked={agreed}
              onChange={(e) => setAgreed(e.target.checked)}
              className="mt-1 h-4 w-4 rounded border-gray-300 text-primary focus:ring-primary"
            />
            <span className="text-xs text-muted-foreground">
              I understand that rankings, scores, and tier designations are analytical reference tools and do not constitute investment advice or trading recommendations.
            </span>
          </label>

          <button
            type="button"
            data-testid="acknowledge-disclaimer-btn"
            disabled={!agreed}
            onClick={handleConfirm}
            className={`w-full py-2.5 px-4 rounded-lg font-medium text-sm flex items-center justify-center gap-2 transition-colors ${
              agreed
                ? 'bg-primary text-primary-foreground hover:bg-primary-700 shadow-sm'
                : 'bg-muted text-muted-foreground cursor-not-allowed opacity-60'
            }`}
          >
            <CheckCircle className="w-4 h-4" aria-hidden="true" />
            <span>Acknowledge and Continue</span>
          </button>
        </div>
      </div>
    </div>
  );
};
