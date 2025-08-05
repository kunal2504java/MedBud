import React from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import { Button } from "./ui/button"
import { ModeToggle } from './ui/mode-toggle';
import { Activity, FileText, Home, Users, BookMarked } from 'lucide-react';
import { DropdownMenu, DropdownMenuTrigger, DropdownMenuContent, DropdownMenuItem, DropdownMenuSeparator } from "@/components/ui/dropdown-menu";

const Header = () => {
  const { user, logout } = useAuth();
  const navigate = useNavigate();

  const handleLogout = () => {
    logout();
    navigate('/');
  };
  return (
    <header className="sticky top-0 z-50 w-full border-b bg-white/80 backdrop-blur-sm dark:bg-zinc-950/80 backdrop:blur-lg">
      <div className="container flex h-16 items-center justify-between px-4 md:px-6">
        <Link to="/" className="flex items-center gap-2 text-lg font-semibold transition-colors hover:text-blue-600">
          <Activity className="h-6 w-6 text-blue-600" />
          <span>MediBud AI</span>
        </Link>
        
        <nav className="md:flex items-center gap-6">
          <Link to="/" className="text-sm font-medium transition-colors hover:text-blue-600 flex items-center gap-2">
            <Home className="h-4 w-4" />
            Home
          </Link>
          <Link to="/upload" className="text-sm font-medium transition-colors hover:text-blue-600 flex items-center gap-2">
            <FileText className="h-4 w-4" />
            Analysis
          </Link>
          <Link to="/doctors" className="text-sm font-medium transition-colors hover:text-blue-600 flex items-center gap-2">
            <Users className="h-4 w-4" />
            Doctors
          </Link>
        </nav>
        
        <div className="flex items-center gap-4">
          <ModeToggle />
          {user ? (
            <>
              <span className="text-sm font-medium hidden md:inline">Welcome, {user.email}</span>
              <DropdownMenu>
                <DropdownMenuTrigger asChild>
                  <Button variant="outline" size="icon">
                    <Activity className="h-5 w-5" />
                  </Button>
                </DropdownMenuTrigger>
                <DropdownMenuContent align="end">
                  <DropdownMenuItem onClick={() => navigate('/records')}>
                    <BookMarked className="mr-2 h-4 w-4" />
                    <span>My Records</span>
                  </DropdownMenuItem>
                  <DropdownMenuSeparator />
                  <DropdownMenuItem onClick={handleLogout}>
                    Logout
                  </DropdownMenuItem>
                </DropdownMenuContent>
              </DropdownMenu>
            </>
          ) : (
            <>
              <Button asChild variant="outline" size="sm">
                <Link to="/login">Login</Link>
              </Button>
              <Button asChild size="sm">
                <Link to="/register">Register</Link>
              </Button>
            </>
          )}
        </div>
      </div>
    </header>
  );
};

export default Header;