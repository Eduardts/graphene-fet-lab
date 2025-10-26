"""
Parameter sweep utility for GFET device characterization
Sweeps gate length, mobility, and quantum capacitance effects
"""
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import pandas as pd
from dataclasses import dataclass
import json


@dataclass
class GFETParameters:
    """GFET device parameters"""
    W: float = 10e-6  # Channel width (m)
    L: float = 1e-6   # Channel length (m)
    Cox: float = 1.15e-8  # Gate oxide capacitance (F/cm^2)
    mu: float = 10000  # Mobility (cm^2/V·s)
    Vdirac: float = 0.0  # Dirac point voltage (V)
    T: float = 300.0  # Temperature (K)
    
    # Quantum capacitance parameters
    enable_quantum_cap: bool = False
    hbar_vF: float = 6.58e-16 * 1e6  # ℏvF in eV·m


class QuantumCapacitanceGFET:
    """GFET model with quantum capacitance effects"""
    
    def __init__(self, params: GFETParameters):
        self.params = params
        self.q = 1.6e-19  # Elementary charge (C)
        self.epsilon_0 = 8.854e-12  # Permittivity of free space
        self.kB = 8.617e-5  # Boltzmann constant (eV/K)
        
    def quantum_capacitance(self, n: float) -> float:
        """
        Calculate quantum capacitance (F/cm^2)
        Cq = 2*e^2/(π*ℏ*vF) * sqrt(π*|n|)
        
        Args:
            n: Carrier density (cm^-2)
        """
        if not self.params.enable_quantum_cap or abs(n) < 1e8:
            return 1e10  # Very large (negligible effect)
        
        # Quantum capacitance per unit area
        n_abs = abs(n) * 1e4  # Convert to m^-2
        Cq = (2 * self.q**2 / (np.pi * self.params.hbar_vF)) * np.sqrt(np.pi * n_abs)
        return Cq * 1e-4  # Convert to F/cm^2
    
    def effective_gate_capacitance(self, n: float) -> float:
        """
        Calculate effective gate capacitance with quantum effects
        1/Ceff = 1/Cox + 1/Cq
        """
        if not self.params.enable_quantum_cap:
            return self.params.Cox
        
        Cq = self.quantum_capacitance(n)
        Ceff = 1.0 / (1.0/self.params.Cox + 1.0/Cq)
        return Ceff
    
    def carrier_density(self, Vgs: float, iterative: bool = True) -> float:
        """
        Calculate carrier density with quantum capacitance
        Requires iterative solution when quantum cap is enabled
        """
        if not self.params.enable_quantum_cap:
            # Simple case without quantum capacitance
            n_gate = self.params.Cox * 1e4 * (Vgs - self.params.Vdirac) / self.q
            n_min = 1e10
            return np.sqrt(n_gate**2 + n_min**2)
        
        # Iterative solution
        n = 1e11  # Initial guess
        for _ in range(10):
            Ceff = self.effective_gate_capacitance(n)
            n_new = Ceff * 1e4 * (Vgs - self.params.Vdirac) / self.q
            n_min = 1e10
            n = np.sqrt(n_new**2 + n_min**2)
        
        return n
    
    def drain_current(self, Vgs: float, Vds: float) -> float:
        """Calculate drain current with quantum capacitance effects"""
        n = self.carrier_density(Vgs)
        
        # Drift-diffusion model
        if abs(Vds) < 0.1:
            # Linear region
            alpha = self.q * self.params.mu * n * 1e-4 * (self.params.W / self.params.L)
            return alpha * Vds
        
        # Velocity saturation
        vF = 1e6  # Fermi velocity (m/s)
        v_sat = vF * np.sqrt(np.pi * n * 1e4) / (1 + abs(Vds)/(2*self.params.L*vF))
        
        Ids = self.params.W * self.q * n * 1e4 * v_sat * np.tanh(Vds / 0.5)
        return Ids


class ParameterSweep:
    """Parameter sweep engine for GFET characterization"""
    
    def __init__(self, base_params: GFETParameters):
        self.base_params = base_params
        self.results = []
        
    def sweep_gate_length(self, L_range: np.ndarray, Vgs_range: np.ndarray, 
                         Vds: float = 1.0) -> pd.DataFrame:
        """
        Sweep gate length and measure characteristics
        
        Args:
            L_range: Array of gate lengths (m)
            Vgs_range: Array of gate voltages (V)
            Vds: Drain-source voltage (V)
        """
        results = []
        
        for L in L_range:
            params = GFETParameters(
                W=self.base_params.W,
                L=L,
                Cox=self.base_params.Cox,
                mu=self.base_params.mu,
                Vdirac=self.base_params.Vdirac,
                enable_quantum_cap=self.base_params.enable_quantum_cap
            )
            
            gfet = QuantumCapacitanceGFET(params)
            
            for Vgs in Vgs_range:
                Ids = gfet.drain_current(Vgs, Vds)
                n = gfet.carrier_density(Vgs)
                
                results.append({
                    'L_um': L * 1e6,
                    'Vgs': Vgs,
                    'Vds': Vds,
                    'Ids_uA': Ids * 1e6,
                    'n_cm2': n,
                    'gm_uS': np.nan  # Will calculate later
                })
        
        df = pd.DataFrame(results)
        
        # Calculate transconductance
        for L in L_range:
            mask = df['L_um'] == L * 1e6
            df.loc[mask, 'gm_uS'] = np.gradient(
                df.loc[mask, 'Ids_uA'].values,
                df.loc[mask, 'Vgs'].values
            )
        
        return df
    
    def sweep_mobility(self, mu_range: np.ndarray, Vgs_range: np.ndarray,
                       Vds: float = 1.0) -> pd.DataFrame:
        """
        Sweep mobility and measure characteristics
        
        Args:
            mu_range: Array of mobility values (cm^2/V·s)
            Vgs_range: Array of gate voltages (V)
            Vds: Drain-source voltage (V)
        """
        results = []
        
        for mu in mu_range:
            params = GFETParameters(
                W=self.base_params.W,
                L=self.base_params.L,
                Cox=self.base_params.Cox,
                mu=mu,
                Vdirac=self.base_params.Vdirac,
                enable_quantum_cap=self.base_params.enable_quantum_cap
            )
            
            gfet = QuantumCapacitanceGFET(params)
            
            for Vgs in Vgs_range:
                Ids = gfet.drain_current(Vgs, Vds)
                
                results.append({
                    'mu': mu,
                    'Vgs': Vgs,
                    'Vds': Vds,
                    'Ids_uA': Ids * 1e6
                })
        
        return pd.DataFrame(results)
    
    def compare_quantum_capacitance(self, Vgs_range: np.ndarray,
                                   Vds: float = 1.0) -> pd.DataFrame:
        """
        Compare results with and without quantum capacitance effects
        """
        results = []
        
        for enable_qc in [False, True]:
            params = GFETParameters(
                W=self.base_params.W,
                L=self.base_params.L,
                Cox=self.base_params.Cox,
                mu=self.base_params.mu,
                Vdirac=self.base_params.Vdirac,
                enable_quantum_cap=enable_qc
            )
            
            gfet = QuantumCapacitanceGFET(params)
            
            for Vgs in Vgs_range:
                Ids = gfet.drain_current(Vgs, Vds)
                n = gfet.carrier_density(Vgs)
                
                if enable_qc:
                    Cq = gfet.quantum_capacitance(n)
                    Ceff = gfet.effective_gate_capacitance(n)
                else:
                    Cq = np.nan
                    Ceff = params.Cox
                
                results.append({
                    'quantum_cap': 'Enabled' if enable_qc else 'Disabled',
                    'Vgs': Vgs,
                    'Ids_uA': Ids * 1e6,
                    'n_cm2': n,
                    'Cq_fF_um2': Cq * 1e15 if not np.isnan(Cq) else np.nan,
                    'Ceff_fF_um2': Ceff * 1e15
                })
        
        return pd.DataFrame(results)
    
    def export_results(self, df: pd.DataFrame, filename: str):
        """Export results to CSV"""
        df.to_csv(filename, index=False)
        print(f"Results saved to {filename}")
    
    def plot_gate_length_sweep(self, df: pd.DataFrame):
        """Plot gate length sweep results"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Transfer characteristics
        for L_um in df['L_um'].unique():
            mask = df['L_um'] == L_um
            axes[0].semilogy(df[mask]['Vgs'], df[mask]['Ids_uA'], 
                           label=f'L = {L_um:.2f} µm', linewidth=2)
        
        axes[0].set_xlabel('Gate Voltage Vgs (V)', fontsize=12)
        axes[0].set_ylabel('Drain Current |Ids| (µA)', fontsize=12)
        axes[0].set_title('Transfer Characteristics vs Gate Length', fontsize=14)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3, which='both')
        
        # Transconductance
        for L_um in df['L_um'].unique():
            mask = df['L_um'] == L_um
            axes[1].plot(df[mask]['Vgs'], df[mask]['gm_uS'],
                        label=f'L = {L_um:.2f} µm', linewidth=2)
        
        axes[1].set_xlabel('Gate Voltage Vgs (V)', fontsize=12)
        axes[1].set_ylabel('Transconductance gm (µS)', fontsize=12)
        axes[1].set_title('Transconductance vs Gate Length', fontsize=14)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('gate_length_sweep.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_mobility_sweep(self, df: pd.DataFrame):
        """Plot mobility sweep results"""
        plt.figure(figsize=(10, 6))
        
        for mu in df['mu'].unique():
            mask = df['mu'] == mu
            plt.semilogy(df[mask]['Vgs'], df[mask]['Ids_uA'],
                        label=f'µ = {mu:.0f} cm²/V·s', linewidth=2)
        
        plt.xlabel('Gate Voltage Vgs (V)', fontsize=12)
        plt.ylabel('Drain Current |Ids| (µA)', fontsize=12)
        plt.title('Transfer Characteristics vs Mobility', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3, which='both')
        plt.tight_layout()
        plt.savefig('mobility_sweep.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_quantum_capacitance_effect(self, df: pd.DataFrame):
        """Plot quantum capacitance comparison"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Current comparison
        for qc in df['quantum_cap'].unique():
            mask = df['quantum_cap'] == qc
            axes[0, 0].semilogy(df[mask]['Vgs'], df[mask]['Ids_uA'],
                              label=qc, linewidth=2)
        
        axes[0, 0].set_xlabel('Gate Voltage Vgs (V)')
        axes[0, 0].set_ylabel('Drain Current |Ids| (µA)')
        axes[0, 0].set_title('Current: Quantum Cap Effect')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3, which='both')
        
        # Carrier density
        for qc in df['quantum_cap'].unique():
            mask = df['quantum_cap'] == qc
            axes[0, 1].plot(df[mask]['Vgs'], df[mask]['n_cm2'],
                          label=qc, linewidth=2)
        
        axes[0, 1].set_xlabel('Gate Voltage Vgs (V)')
        axes[0, 1].set_ylabel('Carrier Density (cm⁻²)')
        axes[0, 1].set_title('Carrier Density: Quantum Cap Effect')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Quantum capacitance
        mask_qc = df['quantum_cap'] == 'Enabled'
        axes[1, 0].plot(df[mask_qc]['Vgs'], df[mask_qc]['Cq_fF_um2'],
                       linewidth=2, color='purple')
        axes[1, 0].set_xlabel('Gate Voltage Vgs (V)')
        axes[1, 0].set_ylabel('Quantum Capacitance (fF/µm²)')
        axes[1, 0].set_title('Quantum Capacitance vs Gate Voltage')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Effective capacitance comparison
        for qc in df['quantum_cap'].unique():
            mask = df['quantum_cap'] == qc
            axes[1, 1].plot(df[mask]['Vgs'], df[mask]['Ceff_fF_um2'],
                          label=qc, linewidth=2)
        
        axes[1, 1].set_xlabel('Gate Voltage Vgs (V)')
        axes[1, 1].set_ylabel('Effective Capacitance (fF/µm²)')
        axes[1, 1].set_title('Effective Gate Capacitance')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('quantum_capacitance_effect.png', dpi=300, bbox_inches='tight')
        plt.show()


def run_complete_sweep_analysis():
    """Run complete parameter sweep analysis"""
    print("=" * 60)
    print("GFET Parameter Sweep Analysis")
    print("=" * 60)
    
    # Base parameters
    base_params = GFETParameters(
        W=10e-6,
        L=1e-6,
        Cox=1.15e-8,
        mu=10000,
        Vdirac=0.0,
        enable_quantum_cap=True
    )
    
    sweep = ParameterSweep(base_params)
    
    # Voltage ranges
    Vgs_range = np.linspace(-2.0, 2.0, 100)
    Vds = 1.0
    
    # 1. Gate length sweep
    print("\n1. Performing gate length sweep...")
    L_range = np.array([0.5e-6, 1.0e-6, 2.0e-6, 5.0e-6])  # 0.5 to 5 µm
    df_length = sweep.sweep_gate_length(L_range, Vgs_range, Vds)
    sweep.export_results(df_length, 'gate_length_sweep.csv')
    sweep.plot_gate_length_sweep(df_length)
    
    # 2. Mobility sweep
    print("\n2. Performing mobility sweep...")
    mu_range = np.array([5000, 10000, 20000, 40000])  # cm^2/V·s
    df_mobility = sweep.sweep_mobility(mu_range, Vgs_range, Vds)
    sweep.export_results(df_mobility, 'mobility_sweep.csv')
    sweep.plot_mobility_sweep(df_mobility)
    
    # 3. Quantum capacitance comparison
    print("\n3. Comparing with/without quantum capacitance...")
    df_quantum = sweep.compare_quantum_capacitance(Vgs_range, Vds)
    sweep.export_results(df_quantum, 'quantum_capacitance_comparison.csv')
    sweep.plot_quantum_capacitance_effect(df_quantum)
    
    print("\n" + "=" * 60)
    print("Analysis complete! Check output files and plots.")
    print("=" * 60)


if __name__ == "__main__":
    run_complete_sweep_analysis()
