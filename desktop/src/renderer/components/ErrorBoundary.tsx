/**
 * Error Boundary Component
 *
 * Catches JavaScript errors anywhere in the child component tree
 */

import { Component, ReactNode } from 'react';
import { AlertTriangle, RefreshCw } from 'lucide-react';

interface Props {
  children: ReactNode;
  fallback?: ReactNode;
}

interface State {
  hasError: boolean;
  error: Error | null;
}

export class ErrorBoundary extends Component<Props, State> {
  constructor(props: Props) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, errorInfo: React.ErrorInfo) {
    console.error('Error caught by boundary:', error, errorInfo);
  }

  handleReset = () => {
    // Clear localStorage to reset state
    localStorage.removeItem('ontology-reasoning-storage');
    this.setState({ hasError: false, error: null });
    window.location.reload();
  };

  render() {
    if (this.state.hasError) {
      if (this.props.fallback) {
        return this.props.fallback;
      }

      return (
        <div className="flex flex-col items-center justify-center min-h-screen bg-dark-950 p-8">
          <div className="bg-dark-900 rounded-2xl border border-dark-700 p-8 max-w-md w-full text-center">
            <div className="p-4 bg-red-500/20 rounded-full w-fit mx-auto mb-6">
              <AlertTriangle className="w-12 h-12 text-red-400" />
            </div>
            <h1 className="text-xl font-semibold text-dark-100 mb-2">
              Something went wrong
            </h1>
            <p className="text-sm text-dark-400 mb-6">
              {this.state.error?.message || 'An unexpected error occurred'}
            </p>
            <button
              onClick={this.handleReset}
              className="flex items-center gap-2 px-6 py-3 mx-auto bg-primary-600 hover:bg-primary-500 text-white rounded-lg font-medium transition-colors"
            >
              <RefreshCw className="w-4 h-4" />
              Reset & Reload
            </button>
            <p className="text-xs text-dark-500 mt-4">
              This will clear all saved settings and reload the app
            </p>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}

export default ErrorBoundary;
