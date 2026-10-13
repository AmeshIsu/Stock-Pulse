import React from 'react';
import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import { TierBadge } from './TierBadge';
import { ConfidencePill } from './ConfidencePill';
import { RiskBadge } from './RiskBadge';
import { ScoreGauge } from './ScoreGauge';
import { FreshnessBanner } from '../layout/FreshnessBanner';

describe('UI Primitives and Accessibility', () => {
  it('renders TierBadge with text and icon for High, Neutral, and Low tiers', () => {
    const { rerender } = render(<TierBadge tier="High" />);
    expect(screen.getByText('High upside')).toBeInTheDocument();
    expect(screen.getByTestId('tier-badge-high')).toBeInTheDocument();

    rerender(<TierBadge tier="Neutral" />);
    expect(screen.getByText('Neutral')).toBeInTheDocument();

    rerender(<TierBadge tier="Low" />);
    expect(screen.getByText('Low upside')).toBeInTheDocument();
  });

  it('renders ConfidencePill strictly using neutral or amber colors, NEVER green', () => {
    render(<ConfidencePill level="Low confidence" />);
    const pill = screen.getByTestId('confidence-pill');
    expect(pill).toBeInTheDocument();
    expect(pill.className).toContain('amber');
    expect(pill.className).not.toContain('emerald');
    expect(pill.className).not.toContain('green');
  });

  it('renders RiskBadge with Normal and Elevated levels', () => {
    const { rerender } = render(<RiskBadge level="Normal" />);
    expect(screen.getByText('Normal risk')).toBeInTheDocument();

    rerender(<RiskBadge level="Elevated" />);
    expect(screen.getByText('Elevated risk')).toBeInTheDocument();
  });

  it('renders ScoreGauge with numerical score bounded between 0 and 100', () => {
    render(<ScoreGauge score={85} tier="High" size="sm" />);
    expect(screen.getByText('85')).toBeInTheDocument();
  });

  it('renders FreshnessBanner with warning text when prices are older than 7 days', () => {
    render(<FreshnessBanner asOfDate="2025-12-01" />);
    expect(screen.getByRole('alert')).toHaveTextContent(
      'Prices were last updated on 2025-12-01'
    );
  });
});
