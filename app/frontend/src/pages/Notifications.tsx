import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { apiClient } from '../lib/api';
import { NotificationItem } from '../lib/types';
import { Bell, CheckCheck, Check, Calendar, ArrowRight } from 'lucide-react';

export const Notifications: React.FC = () => {
  const navigate = useNavigate();

  const [notifications, setNotifications] = useState<NotificationItem[]>([]);
  const [isLoading, setIsLoading] = useState(true);

  const loadNotifications = async () => {
    setIsLoading(true);
    try {
      const data = await apiClient<NotificationItem[]>('/notifications');
      setNotifications(data || []);
    } catch (err) {
      console.error('Failed to load notifications:', err);
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    loadNotifications();
  }, []);

  const handleMarkRead = async (id: string, e: React.MouseEvent) => {
    e.stopPropagation();
    try {
      await apiClient(`/notifications/${id}/read`, { method: 'POST' });
      setNotifications((prev) =>
        prev.map((n) => (n.id === id ? { ...n, is_read: true } : n))
      );
    } catch {
      alert('Could not update notification.');
    }
  };

  const handleMarkAllRead = async () => {
    try {
      await apiClient('/notifications/read-all', { method: 'POST' });
      setNotifications((prev) => prev.map((n) => ({ ...n, is_read: true })));
    } catch {
      alert('Could not mark all as read.');
    }
  };

  return (
    <div className="space-y-6 max-w-4xl mx-auto" data-testid="notifications-page">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-2 border-b">
        <div>
          <h1 className="text-2xl sm:text-3xl font-extrabold tracking-tight text-foreground flex items-center gap-2">
            <Bell className="w-6 h-6 text-primary" />
            <span>Notifications Feed</span>
          </h1>
          <p className="text-xs text-muted-foreground mt-1">
            Historical alerts triggered by daily Colombo Stock Exchange market updates
          </p>
        </div>

        {notifications.some((n) => !n.is_read) && (
          <button
            type="button"
            data-testid="mark-all-read-btn"
            onClick={handleMarkAllRead}
            className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg border border-border bg-card text-xs font-semibold text-foreground hover:bg-muted transition"
          >
            <CheckCheck className="w-4 h-4 text-primary" />
            <span>Mark All as Read</span>
          </button>
        )}
      </div>

      {notifications.length === 0 && !isLoading ? (
        <div className="bg-card border border-border rounded-xl p-12 text-center space-y-3" data-testid="notifications-empty-state">
          <Bell className="w-10 h-10 mx-auto text-muted-foreground opacity-30" />
          <h3 className="font-bold text-foreground">No notifications received yet</h3>
          <p className="text-xs text-muted-foreground max-w-sm mx-auto">
            When market thresholds or tier migrations match your active alerts, they will appear here.
          </p>
        </div>
      ) : (
        <div className="space-y-3" data-testid="notifications-list">
          {notifications.map((notif) => (
            <div
              key={notif.id}
              data-testid={`notification-item-${notif.id}`}
              onClick={() => navigate(`/stocks/${notif.stock_code}`)}
              className={`p-4 rounded-xl border transition cursor-pointer flex flex-col sm:flex-row sm:items-center justify-between gap-3 ${
                notif.is_read
                  ? 'bg-card border-border hover:bg-muted/30 text-muted-foreground'
                  : 'bg-primary/5 border-primary/30 hover:bg-primary/10 text-foreground font-medium shadow-xs'
              }`}
            >
              <div className="space-y-1">
                <div className="flex items-center gap-2">
                  <span className="font-bold text-primary text-sm">{notif.stock_code}</span>
                  <span className="text-xs font-semibold text-foreground">{notif.title}</span>
                  {!notif.is_read && (
                    <span className="w-2 h-2 rounded-full bg-primary" />
                  )}
                </div>
                <p className="text-xs text-muted-foreground">{notif.message}</p>
                <div className="flex items-center gap-2 text-[10px] text-muted-foreground/80 pt-1">
                  <Calendar className="w-3 h-3" />
                  <span>{new Date(notif.created_at).toLocaleDateString()}</span>
                </div>
              </div>

              <div className="flex items-center gap-2 shrink-0 self-end sm:self-center">
                {!notif.is_read && (
                  <button
                    type="button"
                    data-testid={`mark-read-btn-${notif.id}`}
                    onClick={(e) => handleMarkRead(notif.id, e)}
                    className="p-1.5 text-muted-foreground hover:text-primary rounded-lg border bg-card"
                    title="Mark as read"
                    aria-label="Mark notification as read"
                  >
                    <Check className="w-3.5 h-3.5" />
                  </button>
                )}
                <div className="p-1.5 text-muted-foreground flex items-center gap-1 text-xs font-semibold">
                  <span>View Stock</span>
                  <ArrowRight className="w-3.5 h-3.5" />
                </div>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};
