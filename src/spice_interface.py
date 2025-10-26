"""
Python interface for running Verilog-A GFET models via PySpice/Ngspice
"""
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional

try:
    import PySpice
    from PySpice.Spice.Netlist import Circuit
    from PySpice.Unit import *
    PYSPICE_AVAILABLE = True
except ImportError:
    PYSPICE_AVAILABLE = False
    print("PySpice not available. Install with: pip install PySpice")


class GFETSimulator:
    """
    Interface for GFET simulation using SPICE
    Falls back to analytical model if PySpice is not available
    """
    
    def __init__(self, verilog_model_path: str = "../models/graphene_fet.va"):
        """
        Initialize GFET simulator
        
        Args:
            verilog_model_path: Path to Verilog-A model file
        """
        self.model_path = verilog_model_path
        self.use_spice = PYSPICE_AVAILABLE
        
        if not self.use_spice:
            print("Using analytical model as fallback")
    
    def create_circuit(self, W: float = 10e-6, L: float = 1e-6, 
                      mu: float = 10000) -> Optional[object]:
        """
        Create SPICE circuit with GFET device
        
        Args:
            W: Channel width (m)
            L: Channel length (m)
            mu: Mobility (cm^2/V·s)
        
        Returns:
            Circuit object or None if PySpice not available
        """
        if not self.use_spice:
            return None
        
        circuit = Circuit('GFET Test Circuit')
        
        # Voltage sources
        circuit.V('gs', 'gate', circuit.gnd, 0@u_V)
        circuit.V('ds', 'drain', circuit.gnd, 0@u_V)
        
        # GFET device (would need to load Verilog-A model)
        # This is a placeholder - actual implementation needs ngspice with ADMS
        # circuit.X('gfet', 'graphene_fet', 'drain', 'gate', 'source',
        #          W=W, L=L, mu=mu)
        
        return circuit
    
    def dc_sweep(self, Vgs_range: np.ndarray, Vds: float,
                 W: float = 10e-6, L: float = 1e-6,
                 mu: float = 10000) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform DC sweep of Vgs at constant Vds
        
        Args:
            Vgs_range: Array of gate voltages
            Vds: Drain-source voltage
            W: Channel width (m)
            L: Channel length (m)
            mu: Mobility (cm^2/V·s)
        
        Returns:
            Tuple of (Vgs_array, Ids_array)
        """
        if self.use_spice:
            # Would perform actual SPICE simulation here
            # For now, fall back to analytical model
            pass
        
        # Analytical model fallback
        return self._analytical_dc_sweep(Vgs_range, Vds, W, L, mu)
    
    def _analytical_dc_sweep(self, Vgs_range: np.ndarray, Vds: float,
                            W: float, L: float, mu: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Analytical GFET model for DC sweep
        """
        q = 1.6e-19  # Elementary charge
        Cox = 1.15e-8  # Gate oxide capacitance (F/cm^2)
        Vdirac = 0.0  # Dirac point voltage
        
        Ids_array = []
        
        for Vgs in Vgs_range:
            # Carrier density
            n_gate = Cox * 1e4 * (Vgs - Vdirac) / q
            n_min = 1e10
            n = np.sqrt(n_gate**2 + n_min**2)
            
            # Drain current
            if abs(Vds) < 0.1:
                alpha = q * mu * n * 1e-4 * (W / L)
                Ids = alpha * Vds
            else:
                vF = 1e6
                beta = vF * np.sqrt(np.pi * n * 1e4)
                Ids = W * q * n * 1e4 * beta * np.tanh(Vds / 0.5)
            
            Ids_array.append(Ids)
        
        return Vgs_range, np.array(Ids_array)
    
    def plot_transfer_curve(self, Vgs_range: np.ndarray, Vds: float = 1.0,
                           **device_params):
        """
        Plot transfer characteristics (Ids vs Vgs)
        
        Args:
            Vgs_range: Array of gate voltages
            Vds: Drain-source voltage
            **device_params: Device parameters (W, L, mu)
        """
        Vgs, Ids = self.dc_sweep(Vgs_range, Vds, **device_params)
        
        plt.figure(figsize=(10, 6))
        plt.semilogy(Vgs, np.abs(Ids) * 1e6)  # Convert to µA
        plt.xlabel('Gate-Source Voltage Vgs (V)', fontsize=12)
        plt.ylabel('Drain Current |Ids| (µA)', fontsize=12)
        plt.title(f'GFET Transfer Characteristics (Vds = {Vds} V)', 
                 fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3, which='both')
        plt.axvline(x=0, color='red', linestyle='--', alpha=0.5, label='Dirac Point')
        plt.legend()
        plt.tight_layout()
        plt.show()
    
    def plot_output_curves(self, Vds_range: np.ndarray, 
                          Vgs_values: List[float], **device_params):
        """
        Plot output characteristics (Ids vs Vds for different Vgs)
        
        Args:
            Vds_range: Array of drain voltages
            Vgs_values: List of gate voltages
            **device_params: Device parameters (W, L, mu)
        """
        plt.figure(figsize=(10, 6))
        
        for Vgs in Vgs_values:
            Ids_array = []
            for Vds in Vds_range:
                _, Ids = self.dc_sweep(np.array([Vgs]), Vds, **device_params)
                Ids_array.append(Ids[0])
            
            plt.plot(Vds_range, np.array(Ids_array) * 1e6, 
                    label=f'Vgs = {Vgs:.2f} V', linewidth=2)
        
        plt.xlabel('Drain-Source Voltage Vds (V)', fontsize=12)
        plt.ylabel('Drain Current Ids (µA)', fontsize=12)
        plt.title('GFET Output Characteristics', fontsize=14, fontweight='bold')
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


def run_demo():
    """Run demonstration of GFET simulator"""
    print("=== Graphene FET Simulator Demo ===\n")
    
    # Create simulator instance
    sim = GFETSimulator()
    
    # Device parameters
    W = 10e-6  # 10 µm width
    L = 1e-6   # 1 µm length
    mu = 10000  # 10,000 cm^2/V·s mobility
    
    print(f"Device parameters:")
    print(f"  Width: {W*1e6:.1f} µm")
    print(f"  Length: {L*1e6:.1f} µm")
    print(f"  Mobility: {mu} cm²/V·s\n")
    
    # Transfer characteristics
    print("Generating transfer characteristics...")
    Vgs_range = np.linspace(-2.0, 2.0, 200)
    sim.plot_transfer_curve(Vgs_range, Vds=1.0, W=W, L=L, mu=mu)
    
    # Output characteristics
    print("Generating output characteristics...")
    Vds_range = np.linspace(0, 2.0, 100)
    Vgs_values = [-1.0, -0.5, 0.0, 0.5, 1.0]
    sim.plot_output_curves(Vds_range, Vgs_values, W=W, L=L, mu=mu)
    
    print("\nSimulation complete!")


if __name__ == "__main__":
    run_demo()
