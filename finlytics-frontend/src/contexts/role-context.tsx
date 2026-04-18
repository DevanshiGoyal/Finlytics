"use client";

import React, { createContext, useContext, useEffect, useState, ReactNode } from 'react';
import type { Role } from '@/config/navigation';

interface RoleContextType {
  role: Role;
  setRole: (role: Role) => void;
}

const RoleContext = createContext<RoleContextType | undefined>(undefined);

export function RoleProvider({ children }: { children: ReactNode }) {
  const [role, setRoleLocal] = useState<Role>('Risk Analyst');

  useEffect(() => {
    const saved = localStorage.getItem('finlytics-role') as Role | null;
    if (saved && ['Risk Analyst', 'Operations Manager', 'Marketing Team'].includes(saved)) {
      setRoleLocal(saved);
    }
  }, []);

  const setRole = (newRole: Role) => {
    setRoleLocal(newRole);
    localStorage.setItem('finlytics-role', newRole);
  };

  return (
    <RoleContext.Provider value={{ role, setRole }}>
      {children}
    </RoleContext.Provider>
  );
}

export function useRole() {
  const context = useContext(RoleContext);
  if (!context) {
    throw new Error('useRole must be used within RoleProvider');
  }
  return context;
}

