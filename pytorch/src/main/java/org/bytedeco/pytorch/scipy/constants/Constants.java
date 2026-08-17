package org.bytedeco.pytorch.scipy.constants;

/**
 * SciPy constants module equivalent.
 *
 * <p>Physical and mathematical constants with high precision values.
 * References: CODATA 2018, NIST 2019.
 *
 * <p>Usage:
 * <pre>{@code
 * import static org.bytedeco.pytorch.scipy.constants.Constants.*;
 *
 * double c = speed_of_light;        // 299792458 m/s
 * double h = Planck;                // 6.62607015e-34 J s
 * double e = elementary_charge;     // 1.602176634e-19 C
 * }</pre>
 *
 * @see <a href="https://docs.scipy.org/doc/scipy/reference/constants.html">scipy.constants</a>
 */
public final class Constants {

    private Constants() {}

    // =========================================================================
    // Mathematical Constants
    // =========================================================================

    /** Pi (archimedes' constant), 3.14159265358979323846 */
    public static final double pi = 3.141592653589793238462643383279502884;

    /** Pi (alias) */
    public static final double Pi = pi;

    /** 2*pi */
    public static final double pi2 = 2.0 * pi;

    /** pi/2 */
    public static final double pi_2 = pi / 2.0;

    /** 4*pi */
    public static final double pi4 = 4.0 * pi;

    /** pi/4 */
    public static final double pi_4 = pi / 4.0;

    /** pi/180 - conversion from degrees to radians */
    public static final double pi_over_180 = pi / 180.0;

    /** 180/pi - conversion from radians to degrees */
    public static final double _180_over_pi = 180.0 / pi;

    /** 1/pi */
    public static final double _1_over_pi = 1.0 / pi;

    /** 2/pi */
    public static final double _2_over_pi = 2.0 / pi;

    /** sqrt(pi) */
    public static final double sqrt_pi = Math.sqrt(pi);

    /** sqrt(2) */
    public static final double sqrt_2 = Math.sqrt(2.0);

    /** sqrt(2*pi) */
    public static final double sqrt_2pi = Math.sqrt(2.0 * pi);

    /** 1/sqrt(2) */
    public static final double _1_over_sqrt_2 = 1.0 / Math.sqrt(2.0);

    /** 1/sqrt(pi) */
    public static final double _1_over_sqrt_pi = 1.0 / Math.sqrt(pi);

    /** 2/sqrt(pi) */
    public static final double _2_over_sqrt_pi = 2.0 / Math.sqrt(pi);

    /** 2/sqrt(2*pi) */
    public static final double _2_over_sqrt_2pi = 2.0 / Math.sqrt(2.0 * pi);

    /** Euler's number e, 2.71828182845904523536 */
    public static final double e = 2.718281828459045235360287471352662498;

    /** log(2) (natural logarithm of 2) */
    public static final double log2_e = Math.log(2.0);

    /** log(10) (natural logarithm of 10) */
    public static final double log10_e = Math.log(10.0);

    /** log(2) - alias */
    public static final double ln2 = Math.log(2.0);

    /** log(10) - alias */
    public static final double ln10 = Math.log(10.0);

    /** Euler-Mascheroni constant */
    public static final double euler_gamma = 0.5772156649015328606065120900824024310421;

    /** Glaisher-Kinkelin constant */
    public static final double Glaisher = 1.28242712910062263687534256886979;

    /** Khinchin constant */
    public static final double Khinchin = 2.685452001065306445309714835481795;

    /** Twin prime constant */
    public static final double twinprime = 0.660161815846869573927812110014555;

    /** Mertens constant */
    public static final double mertens = 0.2614972128476427837554268386086958590516;

    /** Golden ratio (phi) */
    public static final double golden = 1.618033988749894848204586834365638118;

    /** Golden ratio - alias */
    public static final double golden_ratio = golden;

    /** Silver ratio (1 + sqrt(2)) */
    public static final double silver = 1.0 + Math.sqrt(2.0);

    /** Catalan's constant */
    public static final double catalan = 0.915965594177219015054603514932384110774;

    /** Apery's constant (zeta(3)) */
    public static final double zeta3 = 1.202056903159594285399738161511449990764;

    /** Plastic number */
    public static final double plastic = 1.324717957244746025960908854478097340;

    /** Supergolden ratio */
    public static final double supergolden = 1.46557123187676802665673122521937918;

    /** Pythagorean constant (sqrt(2)) */
    public static final double Pythagoras = Math.sqrt(2.0);

    /** Pythagorean constant - alias */
    public static final double pythagoras = Pythagoras;

    /** Feigenbaum constant (delta) */
    public static final double Feigenbaum = 4.669201609102990;

    /** Feigenbaum constant (alpha) */
    public static final double feigenbaum = 2.502907875095892;

    /** Omega constant (Omega = W(1)) */
    public static final double Omega = 0.56714329040978387299996866221035554975381578718651;

    /** Speed of light in vacuum (c) */
    public static final double c = 299792458.0;

    /** Speed of light - alias */
    public static final double speed_of_light = c;

    /** Speed of light in km/s */
    public static final double c_km_s = c / 1000.0;

    /** Speed of light in miles/s */
    public static final double c_miles_s = c / 1609.344;

    /** Planck constant (h) */
    public static final double h = 6.62607015e-34;

    /** Planck constant - alias */
    public static final double Planck = h;

    /** Reduced Planck constant (hbar) */
    public static final double hbar = 1.054571817e-34;

    /** Reduced Planck constant - alias */
    public static final double Planck_constant = hbar;

    /** Gravitational constant (G) */
    public static final double G = 6.67430e-11;

    /** Gravitational constant - alias */
    public static final double gravitational_constant = G;

    /** Newtonian constant - alias */
    public static final double newtonian_constant_of_gravitation = G;

    /** Boltzmann constant (k_B) */
    public static final double k_B = 1.380649e-23;

    /** Boltzmann constant - alias */
    public static final double Boltzmann = k_B;

    /** Boltzmann constant in eV/K */
    public static final double k_B_in_eV = k_B / 1.602176634e-19;

    /** Elementary charge (e) */
    public static final double e_charge = 1.602176634e-19;

    /** Elementary charge - alias */
    public static final double elementary_charge = e_charge;

    /** 1/e_charge */
    public static final double _1_over_e_charge = 1.0 / e_charge;

    /** Avogadro constant (N_A) */
    public static final double N_A = 6.02214076e23;

    /** Avogadro constant - alias */
    public static final double Avogadro = N_A;

    /** Avogadro constant - alias */
    public static final double avogadro = N_A;

    /** Gas constant (R) */
    public static final double R = 8.314462618;

    /** Gas constant - alias */
    public static final double gas_constant = R;

    /** Gas constant - alias */
    public static final double molar_gas_constant = R;

    /** Faraday constant (F) */
    public static final double F = 96485.33212;

    /** Faraday constant - alias */
    public static final double Faraday = F;

    /** Faraday constant - alias */
    public static final double faraday_constant = F;

    /** Stefan-Boltzmann constant */
    public static final double sigma = 5.670374419e-8;

    /** Stefan-Boltzmann constant - alias */
    public static final double Stefan_Boltzmann = sigma;

    /** Stefan-Boltzmann constant - alias */
    public static final double stefan_boltzmann = sigma;

    /** Wien displacement law constant */
    public static final double b = 2.897771955e-3;

    /** Wien displacement law constant - alias */
    public static final double Wien = b;

    /** Wien displacement law constant - alias */
    public static final double wien_displacement_law_constant = b;

    /** Rydberg constant */
    public static final double R_inf = 10973731.568160;

    /** Rydberg constant - alias */
    public static final double Rydberg = R_inf;

    /** Rydberg constant - alias */
    public static final double rydberg = R_inf;

    /** Electron mass */
    public static final double m_e = 9.1093837015e-31;

    /** Electron mass - alias */
    public static final double electron_mass = m_e;

    /** Proton mass */
    public static final double m_p = 1.67262192369e-27;

    /** Proton mass - alias */
    public static final double proton_mass = m_p;

    /** Neutron mass */
    public static final double m_n = 1.67492749804e-27;

    /** Neutron mass - alias */
    public static final double neutron_mass = m_n;

    /** Muon mass */
    public static final double m_mu = 1.883531627e-28;

    /** Muon mass - alias */
    public static final double muon_mass = m_mu;

    /** Tau mass */
    public static final double m_tau = 3.16754e-27;

    /** Tau mass - alias */
    public static final double tau_mass = m_tau;

    /** Atomic mass constant (u) */
    public static final double u = 1.66053906660e-27;

    /** Atomic mass constant - alias */
    public static final double atomic_mass = u;

    /** Atomic mass constant - alias */
    public static final double unified_atomic_mass_unit = u;

    /** Top quark mass */
    public static final double m_t = 3.0784e-25;

    /** Top quark mass - alias */
    public static final double top_quark_mass = m_t;

    /** Bohr magneton */
    public static final double mu_B = 9.2740100783e-24;

    /** Bohr magneton - alias */
    public static final double Bohr_magneton = mu_B;

    /** Nuclear magneton */
    public static final double mu_N = 5.050783746e-27;

    /** Nuclear magneton - alias */
    public static final double nuclear_magneton = mu_N;

    /** Magnetic flux quantum */
    public static final double phi_0 = 2.067833848e-15;

    /** Magnetic flux quantum - alias */
    public static final double magnetic_flux_quantum = phi_0;

    /** Conductance quantum */
    public static final double G_0 = 7.748091729e-5;

    /** Conductance quantum - alias */
    public static final double conductance_quantum = G_0;

    /** Von Klitzing constant */
    public static final double R_k = 25812.80745;

    /** Von Klitzing constant - alias */
    public static final double von_klitzing_constant = R_k;

    /** Josephson constant */
    public static final double K_J = 4.835978484e14;

    /** Josephson constant - alias */
    public static final double josephson_constant = K_J;

    /** Vacuum permeability */
    public static final double mu_0 = 1.25663706212e-6;

    /** Vacuum permeability - alias */
    public static final double vacuum_permeability = mu_0;

    /** Vacuum permittivity */
    public static final double epsilon_0 = 8.8541878128e-12;

    /** Vacuum permittivity - alias */
    public static final double vacuum_permittivity = epsilon_0;

    /** Classical electron radius */
    public static final double r_e = 2.8179403262e-15;

    /** Classical electron radius - alias */
    public static final double classical_electron_radius = r_e;

    /** Thomson cross section */
    public static final double sigma_e = 6.6524587321e-29;

    /** Thomson cross section - alias */
    public static final double thomson_cross_section = sigma_e;

    /** Bohr radius */
    public static final double a_0 = 5.29177210903e-11;

    /** Bohr radius - alias */
    public static final double Bohr_radius = a_0;

    /** Hartree energy */
    public static final double E_h = 4.3597447222071e-18;

    /** Hartree energy - alias */
    public static final double Hartree_energy = E_h;

    /** Hartree energy in eV */
    public static final double Hartree_in_eV = 27.211386245988;

    /** Fine-structure constant */
    public static final double alpha = 7.2973525693e-3;

    /** Fine-structure constant - alias */
    public static final double fine_structure = alpha;

    /** Fine-structure constant - alias */
    public static final double alpha_el = alpha;

    /** Inverse fine-structure constant */
    public static final double alpha_inv = 137.035999084;

    /** Inverse fine-structure constant - alias */
    public static final double inverse_fine_structure = alpha_inv;

    /** Compton wavelength (electron) */
    public static final double lambda_C = 2.42631023867e-12;

    /** Compton wavelength (electron) - alias */
    public static final double compton_wavelength = lambda_C;

    /** Reduced Compton wavelength */
    public static final double lambda_C_bar = 3.8615926796e-13;

    /** Reduced Compton wavelength - alias */
    public static final double reduced_Compton_wavelength = lambda_C_bar;

    /** Atomic unit of length */
    public static final double a_0_au = 5.29177210903e-11;

    /** Atomic unit of time */
    public static final double t_au = 2.4188843265857e-17;

    /** Atomic unit of energy */
    public static final double E_au = 4.3597447222071e-18;

    /** Atomic unit of velocity */
    public static final double v_au = 2.18769126364e6;

    /** Electric constant */
    public static final double electric_constant = epsilon_0;

    /** Magnetic constant */
    public static final double magnetic_constant = mu_0;

    /** Characteristic impedance of vacuum */
    public static final double Z_0 = 376.730313668;

    /** Characteristic impedance of vacuum - alias */
    public static final double impedance_of_free_space = Z_0;

    /** Newtonian constant of gravitation in m^3/(kg s^2) */
    public static final double G_newtonian = G;

    /** Conversion constant from kg to lbs */
    public static final double kg_pounds = 2.2046226218487757;

    /** Conversion constant from lbs to kg */
    public static final double pound_kg = 0.45359237;

    /** Conversion constant from kg to ounces */
    public static final double kg_ounces = 35.27396195;

    /** Conversion constant from ounce to kg */
    public static final double ounce_kg = 0.028349523125;

    /** Conversion constant from kg to grams */
    public static final double kg_grams = 1000.0;

    /** Conversion constant from gram to kg */
    public static final double gram_kg = 0.001;

    /** Conversion constant from kg to tonnes (metric tons) */
    public static final double kg_tonne = 0.001;

    /** Conversion constant from tonne to kg */
    public static final double tonne_kg = 1000.0;

    /** Conversion constant from kg to carat */
    public static final double kg_carat = 5000.0;

    /** Conversion constant from carat to kg */
    public static final double carat_kg = 0.0002;

    /** Conversion constant from kg to grain */
    public static final double kg_grain = 15432.358353;

    /** Conversion constant from grain to kg */
    public static final double grain_kg = 0.00006479891;

    /** Angstrom in meters */
    public static final double angstrom = 1e-10;

    /** Inch in meters */
    public static final double inch = 0.0254;

    /** Inch - alias */
    public static final double Int_inch = inch;

    /** Foot in meters */
    public static final double foot = 0.3048;

    /** Foot - alias */
    public static final double Int_foot = foot;

    /** Yard in meters */
    public static final double yard = 0.9144;

    /** Yard - alias */
    public static final double Int_yard = yard;

    /** Mile in meters */
    public static final double mile = 1609.344;

    /** Mile - alias */
    public static final double Int_mile = mile;

    /** Mil in meters */
    public static final double mil = 0.0000254;

    /** Microinch in meters */
    public static final double microinch = 2.54e-8;

    /** Hand in meters */
    public static final double hand = 0.1016;

    /** Rod in meters */
    public static final double rod = 5.0292;

    /** Fathom in meters */
    public static final double fathom = 1.8288;

    /** Nautical mile in meters */
    public static final double nautical_mile = 1852.0;

    /** Nautical mile - alias */
    public static final double Int_nautical_mile = nautical_mile;

    /** Cable length in meters */
    public static final double cable_length = 185.2;

    /** Furlong in meters */
    public static final double furlong = 201.168;

    /** League in meters */
    public static final double league = 4828.032;

    /** Link in meters */
    public static final double link = 0.201168;

    /** Chain in meters */
    public static final double chain = 20.1168;

    /** Point in meters */
    public static final double point = 0.0003527778;

    /** Pica in meters */
    public static final double pica = 0.0042333333;

    /** Astronomical unit in meters */
    public static final double au = 1.495978707e11;

    /** Astronomical unit - alias */
    public static final double astronomical_unit = au;

    /** Light year in meters */
    public static final double lightyear = 9.4607304725808e15;

    /** Light year - alias */
    public static final double light_year = lightyear;

    /** Parsec in meters */
    public static final double parsec = 3.0856775814913673e16;

    /** Parsec - alias */
    public static final double pc = parsec;

    /** Kiloparsec in meters */
    public static final double kpc = 1000.0 * parsec;

    /** Megaparsec in meters */
    public static final double Mpc = 1e6 * parsec;

    /** Gigaparsec in meters */
    public static final double Gpc = 1e9 * parsec;

    /** Solar radius in meters */
    public static final double R_sun = 6.957e8;

    /** Solar radius - alias */
    public static final double solar_radius = R_sun;

    /** Solar mass in kg */
    public static final double M_sun = 1.98892e30;

    /** Solar mass - alias */
    public static final double solar_mass = M_sun;

    /** Solar luminosity in watts */
    public static final double L_sun = 3.828e26;

    /** Solar luminosity - alias */
    public static final double solar_luminosity = L_sun;

    /** Earth mass in kg */
    public static final double M_earth = 5.97217e24;

    /** Earth mass - alias */
    public static final double earth_mass = M_earth;

    /** Earth radius in meters */
    public static final double R_earth = 6.3781e6;

    /** Earth radius - alias */
    public static final double earth_radius = R_earth;

    /** Jupiter mass in kg */
    public static final double M_jup = 1.89813e27;

    /** Jupiter mass - alias */
    public static final double jupiter_mass = M_jup;

    /** Jupiter radius in meters */
    public static final double R_jup = 6.9911e7;

    /** Jupiter radius - alias */
    public static final double jupiter_radius = R_jup;

    /** Lunar mass in kg */
    public static final double M_luna = 7.342e22;

    /** Lunar mass - alias */
    public static final double lunar_mass = M_luna;

    /** Lunar radius in meters */
    public static final double R_luna = 1.7371e6;

    /** Lunar radius - alias */
    public static final double lunar_radius = R_luna;

    /** Sidereal year in seconds */
    public static final double year = 31558149.7635456;

    /** Julian year in seconds */
    public static final double jyear = 31557600.0;

    /** Tropical year in seconds */
    public static final double tyear = 31556925.190;

    /** Day in seconds */
    public static final double day = 86400.0;

    /** Day - alias */
    public static final double Day = day;

    /** Week in seconds */
    public static final double week = 7.0 * day;

    /** Hour in seconds */
    public static final double hour = 3600.0;

    /** Hour - alias */
    public static final double Hour = hour;

    /** Minute in seconds */
    public static final double minute = 60.0;

    /** Minute - alias */
    public static final double Minute = minute;

    /** Second in seconds (1.0) */
    public static final double second = 1.0;

    /** Millisecond in seconds */
    public static final double millisecond = 1e-3;

    /** Microsecond in seconds */
    public static final double microsecond = 1e-6;

    /** Nanosecond in seconds */
    public static final double nanosecond = 1e-9;

    /** Picosecond in seconds */
    public static final double picosecond = 1e-12;

    /** Femtosecond in seconds */
    public static final double femtosecond = 1e-15;

    /** Attosecond in seconds */
    public static final double attosecond = 1e-18;

    /** Hectare in square meters */
    public static final double hectare = 10000.0;

    /** Hectare - alias */
    public static final double Int_hectare = hectare;

    /** Acre in square meters */
    public static final double acre = 4046.8564224;

    /** Acre - alias */
    public static final double Int_acre = acre;

    /** Square meter */
    public static final double sq_meter = 1.0;

    /** Square kilometer */
    public static final double sq_km = 1e6;

    /** Square centimeter */
    public static final double sq_cm = 1e-4;

    /** Square millimeter */
    public static final double sq_mm = 1e-6;

    /** Square mile */
    public static final double sq_mile = mile * mile;

    /** Square foot */
    public static final double sq_foot = foot * foot;

    /** Square inch */
    public static final double sq_inch = inch * inch;

    /** Barn in square meters */
    public static final double barn = 1e-28;

    /** Cubic meter */
    public static final double cubic_meter = 1.0;

    /** Cubic kilometer */
    public static final double cubic_km = 1e9;

    /** Cubic centimeter (cc) */
    public static final double cm3 = 1e-6;

    /** Cubic foot */
    public static final double cubic_foot = foot * foot * foot;

    /** Cubic inch */
    public static final double cubic_inch = inch * inch * inch;

    /** Liter in cubic meters */
    public static final double liter = 1e-3;

    /** Liter - alias */
    public static final double litre = liter;

    /** Milliliter in cubic meters */
    public static final double milliliter = 1e-6;

    /** Milliliter - alias */
    public static final double millilitre = milliliter;

    /** Gallon (US) in cubic meters */
    public static final double gallon = 0.003785411784;

    /** Gallon (US) - alias */
    public static final double gallon_US = gallon;

    /** Gallon (UK) in cubic meters */
    public static final double gallon_imp = 0.00454609;

    /** Quart (US) in cubic meters */
    public static final double quart = 0.000946352946;

    /** Pint (US) in cubic meters */
    public static final double pint = 0.000473176473;

    /** Cup (US) in cubic meters */
    public static final double cup = 0.0002365882365;

    /** Fluid ounce (US) in cubic meters */
    public static final double fluid_ounce = 2.95735296875e-5;

    /** Fluid ounce (UK) in cubic meters */
    public static final double fluid_ounce_imp = 2.84130625e-5;

    /** Teaspoon (US) in cubic meters */
    public static final double teaspoon = 4.92892159375e-6;

    /** Tablespoon (US) in cubic meters */
    public static final double tablespoon = 1.478676478125e-5;

    /** Barrel in cubic meters */
    public static final double barrel = 0.158987294928;

    /** Knot in meters/second */
    public static final double knot = nautical_mile / hour;

    /** Knot - alias */
    public static final double Int_knot = knot;

    /** Mach number in meters/second */
    public static final double mach = 340.5;

    /** Mach number - alias */
    public static final double speed_of_sound = mach;

    /** Mile per hour in meters/second */
    public static final double mph = mile / hour;

    /** Kilometer per hour in meters/second */
    public static final double kph = 1000.0 / hour;

    /** Foot per second in meters/second */
    public static final double fps = foot / 1.0;

    /** Standard atmosphere in pascals */
    public static final double atm = 101325.0;

    /** Standard atmosphere - alias */
    public static final double atmosphere = atm;

    /** Torr in pascals */
    public static final double torr = atm / 760.0;

    /** Millimeter of mercury in pascals */
    public static final double mmHg = 133.322387415;

    /** Centimeter of mercury in pascals */
    public static final double cmHg = 10.0 * mmHg;

    /** Inch of mercury in pascals */
    public static final double inHg = 25.4 * mmHg;

    /** Bar in pascals */
    public static final double bar = 100000.0;

    /** Millibar in pascals */
    public static final double millibar = 100.0;

    /** Microbar in pascals */
    public static final double microbar = 0.1;

    /** PSI in pascals */
    public static final double psi = 6894.757293168;

    /** Technical atmosphere in pascals */
    public static final double at = 98066.5;

    /** Pound-force per square inch - alias */
    public static final double lbf_in2 = psi;

    /** Electron volt in joules */
    public static final double eV = 1.602176634e-19;

    /** Electron volt - alias */
    public static final double electron_volt = eV;

    /** Kilo electron volt */
    public static final double keV = 1000.0 * eV;

    /** Mega electron volt */
    public static final double MeV = 1e6 * eV;

    /** Giga electron volt */
    public static final double GeV = 1e9 * eV;

    /** Tera electron volt */
    public static final double TeV = 1e12 * eV;

    /** Rydberg in joules */
    public static final double Ry = 2.1798723611035e-18;

    /** Rydberg in eV */
    public static final double Ry_in_eV = 13.605693122994;

    /** Calorie (thermochemical) in joules */
    public static final double calorie = 4.184;

    /** Calorie - alias */
    public static final double cal = calorie;

    /** Kilocalorie in joules */
    public static final double kilocalorie = 4184.0;

    /** Food calorie (alias) */
    public static final double Calorie = kilocalorie;

    /** BTU in joules */
    public static final double btu = 1055.05585262;

    /** BTU - alias */
    public static final double BTU = btu;

    /** Erg in joules */
    public static final double erg = 1e-7;

    /** Foot-pound in joules */
    public static final double foot_pound = 1.35581794833;

    /** Foot-pound - alias */
    public static final double ft_lbf = foot_pound;

    /** Ton of TNT in joules */
    public static final double ton_TNT = 4.184e9;

    /** Horsepower in watts */
    public static final double horsepower = 745.6998715822702;

    /** Horsepower - alias */
    public static final double hp = horsepower;

    /** Dyne in newtons */
    public static final double dyne = 1e-5;

    /** Pound-force in newtons */
    public static final double pound_force = 4.4482216152605;

    /** Pound-force - alias */
    public static final double lbf = pound_force;

    /** Kilogram-force in newtons */
    public static final double kgf = 9.80665;

    /** Ounce-force in newtons */
    public static final double ounce_force = 0.27801385095378125;

    /** Newton-meter in joules */
    public static final double nm_to_j = 1.0;

    /** Degree in radians */
    public static final double degree = pi / 180.0;

    /** Degree - alias */
    public static final double deg = degree;

    /** Arc-minute in radians */
    public static final double arcmin = degree / 60.0;

    /** Arc-minute - alias */
    public static final double arc_minute = arcmin;

    /** Arc-minute - alias */
    public static final double arcminute = arcmin;

    /** Arc-second in radians */
    public static final double arcsec = degree / 3600.0;

    /** Arc-second - alias */
    public static final double arc_second = arcsec;

    /** Arc-second - alias */
    public static final double arcsecond = arcsec;

    /** Steradian */
    public static final double steradian = 1.0;

    /** Radian */
    public static final double radian = 1.0;

    /** Radian - alias */
    public static final double rad = radian;

    /** Molar volume of ideal gas at STP in m^3/mol */
    public static final double molar_volume = 22.41396954e-3;

    /** Loschmidt constant at STP in m^-3 */
    public static final double Loschmidt = 2.6867811e25;

    /** Loschmidt constant - alias */
    public static final double loschmidt = Loschmidt;

    /** Density of water at 4degC in kg/m^3 */
    public static final double density_water = 999.97495;

    /** Density of mercury in kg/m^3 */
    public static final double density_mercury = 13595.1;

    /** Density of air at STP in kg/m^3 */
    public static final double density_air = 1.225;

    /** Density of seawater in kg/m^3 */
    public static final double density_seawater = 1025.0;

    /** Density of iron in kg/m^3 */
    public static final double density_iron = 7874.0;

    /** Density of gold in kg/m^3 */
    public static final double density_gold = 19320.0;

    /** Density of silver in kg/m^3 */
    public static final double density_silver = 10490.0;

    /** Density of copper in kg/m^3 */
    public static final double density_copper = 8960.0;

    /** Gravity of Earth in m/s^2 */
    public static final double g = 9.80665;

    /** Gravity of Earth - alias */
    public static final double gravity = g;

    /** Standard gravity - alias */
    public static final double standard_gravity = g;

    /** Gravity of Moon in m/s^2 */
    public static final double g_moon = 1.6249806;

    /** Gravity of Moon - alias */
    public static final double moon_gravity = g_moon;

    /** Gravity of Mars in m/s^2 */
    public static final double g_mars = 3.710;

    /** Gravity of Mars - alias */
    public static final double mars_gravity = g_mars;

    /** Gravity of Jupiter in m/s^2 */
    public static final double g_jupiter = 24.79;

    /** Gravity of Jupiter - alias */
    public static final double jupiter_gravity = g_jupiter;

    /** Triple point of water in K */
    public static final double TTP = 273.16;

    /** Triple point of water - alias */
    public static final double triple_point = TTP;

    /** Boiling point of water at 1 atm in K */
    public static final double T_bp = 373.15;

    /** Boiling point - alias */
    public static final double boiling_point = T_bp;

    /** Freezing point of water at 1 atm in K */
    public static final double T_fp = 273.15;

    /** Freezing point - alias */
    public static final double freezing_point = T_fp;

    /** Zero Celsius in K */
    public static final double zero_Celsius = 273.15;

    /** Zero Celsius - alias */
    public static final double zero_Celsius_alt = 273.15;

    // =========================================================================
    // Conversion factors (compound)
    // =========================================================================

    /** Conversion factor from km/h to m/s */
    public static final double kmh_to_ms = 1000.0 / 3600.0;

    /** Conversion factor from mph to m/s */
    public static final double mph_to_ms = mile / hour;

    /** Conversion factor from knots to m/s */
    public static final double knot_to_ms = knot;

    /** Conversion factor from BTU/h to watts */
    public static final double btu_h_to_watts = btu / hour;

    /** Conversion factor from HP to watts */
    public static final double hp_to_watts = horsepower;

    /** Conversion factor from cal/s to watts */
    public static final double cal_s_to_watts = calorie;

    /** Conversion factor from erg/s to watts */
    public static final double erg_s_to_watts = 1e-7;

    /** Conversion factor from ton (refrigeration) to watts */
    public static final double ton_cooling = 3516.8528421;

    /** Standard room temperature in K */
    public static final double T_room = 293.15;

    /** Standard room temperature in Celsius */
    public static final double T_room_C = 20.0;

    /** Standard body temperature in K */
    public static final double T_body = 310.15;

    /** Standard body temperature in Celsius */
    public static final double T_body_C = 37.0;

    /** Surface temperature of the Sun in K */
    public static final double T_sun = 5778.0;

    /** Surface temperature of the Sun - alias */
    public static final double sun_temperature = T_sun;

    /** Conversion factor from gauss to tesla */
    public static final double gauss_to_tesla = 1e-4;

    /** Conversion factor from tesla to gauss */
    public static final double tesla_to_gauss = 1e4;

    /** Conversion factor from ampere to statampere */
    public static final double ampere_to_statampere = 2.99792458e9;

    /** Conversion factor from ampere to abampere */
    public static final double ampere_to_abampere = 0.1;

    /** Conversion factor from coulomb to statcoulomb */
    public static final double coulomb_to_statcoulomb = 2.99792458e9;

    /** Conversion factor from coulomb to abcoulomb */
    public static final double coulomb_to_abcoulomb = 0.1;

    /** Conversion factor from volt to statvolt */
    public static final double volt_to_statvolt = 2.99792458e2;

    /** Conversion factor from volt to abvolt */
    public static final double volt_to_abvolt = 1e8;

    /** Conversion factor from ohm to statohm */
    public static final double ohm_to_statohm = 8.98755179e11;

    /** Conversion factor from ohm to abohm */
    public static final double ohm_to_abohm = 1e-9;

    // =========================================================================
    // Lookup Utility
    // =========================================================================

    /**
     * Look up a constant by name.
     *
     * @param name the constant name (e.g. "speed_of_light", "c", "Planck")
     * @return the value of the constant, or NaN if not found
     */
    public static double lookup(String name) {
        if (name == null) return Double.NaN;
        switch (name.toLowerCase().replace("-", "_").replace(" ", "_")) {
            // Mathematical
            case "pi": case "π": return pi;
            case "e": return e;
            case "euler_gamma": case "euler-gamma": return euler_gamma;
            case "golden": case "golden_ratio": return golden;
            case "silver": return silver;
            case "catalan": return catalan;
            case "glaisher": return Glaisher;
            case "khinchin": return Khinchin;
            case "twinprime": case "twin_prime": return twinprime;
            case "mertens": return mertens;
            case "omega": return Omega;
            case "feigenbaum": case "delta": return Feigenbaum;
            case "alpha_f": return feigenbaum;

            // Physical
            case "c": case "speed_of_light": return c;
            case "h": case "planck": return h;
            case "hbar": case "reduced_planck": return hbar;
            case "g": case "gravitational_constant": return G;
            case "k": case "k_b": case "boltzmann": return k_B;
            case "e_charge": case "elementary_charge": return e_charge;
            case "n_a": case "avogadro": return N_A;
            case "r": case "gas_constant": return R;
            case "f": case "faraday": return F;
            case "sigma": case "stefan_boltzmann": return sigma;
            case "wien": case "wien_displacement": return b;
            case "rydberg": case "r_inf": return R_inf;
            case "m_e": case "electron_mass": return m_e;
            case "m_p": case "proton_mass": return m_p;
            case "m_n": case "neutron_mass": return m_n;
            case "u": case "atomic_mass": return u;
            case "mu_b": case "bohr_magneton": return mu_B;
            case "mu_n": case "nuclear_magneton": return mu_N;
            case "phi_0": case "mag_flux_quantum": return phi_0;
            case "g_0": case "conductance_quantum": return G_0;
            case "mu_0": case "vacuum_permeability": return mu_0;
            case "epsilon_0": case "vacuum_permittivity": return epsilon_0;
            case "a_0": case "bohr_radius": return a_0;
            case "alpha": case "fine_structure": return alpha;
            case "e_h": case "hartree": return E_h;

            // Time
            case "second": case "s": return second;
            case "minute": case "min": return minute;
            case "hour": case "h_time": return hour;
            case "day": return day;
            case "week": return week;
            case "year": return year;
            case "jyear": case "julian_year": return jyear;

            // Length
            case "meter": case "m": return 1.0;
            case "km": case "kilometer": return 1000.0;
            case "cm": case "centimeter": return 0.01;
            case "mm": case "millimeter": return 0.001;
            case "inch": case "in": return inch;
            case "foot": case "ft": return foot;
            case "yard": case "yd": return yard;
            case "mile": case "mi": return mile;
            case "angstrom": return angstrom;
            case "au": case "astronomical_unit": return au;
            case "lightyear": case "ly": return lightyear;
            case "parsec": case "pc": return parsec;

            // Mass
            case "kg": case "kilogram": return 1.0;
            case "gram": case "g_mass": return gram_kg;
            case "pound": case "lb": return pound_kg;
            case "ounce": case "oz": return ounce_kg;
            case "ton": case "tonne": return tonne_kg;
            case "carat": return carat_kg;
            case "grain": return grain_kg;

            // Volume
            case "liter": case "l": return liter;
            case "milliliter": case "ml": return milliliter;
            case "gallon": return gallon;
            case "quart": return quart;
            case "pint": return pint;
            case "cup": return cup;
            case "fluid_ounce": return fluid_ounce;
            case "teaspoon": return teaspoon;
            case "tablespoon": return tablespoon;

            // Pressure
            case "pascal": case "pa": return 1.0;
            case "kpa": case "kilopascal": return 1000.0;
            case "mpa": case "megapascal": return 1e6;
            case "bar": return bar;
            case "millibar": case "mbar": return millibar;
            case "atm": case "atmosphere": return atm;
            case "torr": return torr;
            case "mmhg": return mmHg;
            case "psi": return psi;

            // Energy
            case "joule": case "j": return 1.0;
            case "kilojoule": case "kj": return 1000.0;
            case "megajoule": case "mj": return 1e6;
            case "cal": case "calorie": return calorie;
            case "kcal": case "kilocalorie": return kilocalorie;
            case "btu": return btu;
            case "erg": return erg;
            case "ev": case "electron_volt": return eV;
            case "kev": return keV;
            case "mev": return MeV;
            case "gev": return GeV;

            // Power
            case "watt": case "w": return 1.0;
            case "kilowatt": case "kw": return 1000.0;
            case "megawatt": case "mw": return 1e6;
            case "horsepower": case "hp": return horsepower;

            // Force
            case "newton": case "n": return 1.0;
            case "kilonewton": case "kn_force": return 1000.0;
            case "dyne": return dyne;
            case "lbf": case "pound_force": return pound_force;
            case "kgf": case "kilogram_force": return kgf;

            // Temperature
            case "kelvin": case "k_temp": return 1.0;
            case "celsius": case "°c": return 1.0;
            case "fahrenheit": case "°f": return 5.0 / 9.0;
            case "rankine": case "°r": return 5.0 / 9.0;

            // Frequency
            case "hertz": case "hz": return 1.0;
            case "khz": case "kilohertz": return 1000.0;
            case "mhz": case "megahertz": return 1e6;
            case "ghz": case "gigahertz": return 1e9;

            // Angle
            case "radian": case "rad": return 1.0;
            case "degree": case "deg": return degree;
            case "arcmin": case "arcminute": return arcmin;
            case "arcsec": case "arcsecond": return arcsec;

            default: return Double.NaN;
        }
    }

    /**
     * Get all available constant names.
     */
    public static String[] names() {
        return new String[]{
            "pi", "e", "euler_gamma", "golden", "silver", "catalan", "Glaisher", "Khinchin",
            "twinprime", "mertens", "Omega", "Feigenbaum", "feigenbaum",
            "c", "speed_of_light", "h", "Planck", "hbar", "G", "gravitational_constant",
            "k_B", "Boltzmann", "e_charge", "elementary_charge", "N_A", "Avogadro",
            "R", "gas_constant", "F", "Faraday", "sigma", "Stefan_Boltzmann",
            "Rydberg", "R_inf", "m_e", "m_p", "m_n", "u", "atomic_mass",
            "mu_B", "Bohr_magneton", "mu_N", "nuclear_magneton",
            "phi_0", "magnetic_flux_quantum", "G_0", "conductance_quantum",
            "mu_0", "vacuum_permeability", "epsilon_0", "vacuum_permittivity",
            "a_0", "Bohr_radius", "alpha", "fine_structure", "E_h", "Hartree_energy",
            "alpha_inv", "inverse_fine_structure",
            "lambda_C", "compton_wavelength", "sigma_e", "thomson_cross_section",
            "r_e", "classical_electron_radius",
            "year", "jyear", "day", "hour", "minute", "second",
            "angstrom", "au", "lightyear", "parsec",
            "hectare", "acre", "liter", "gallon", "knot",
            "atm", "torr", "mmHg", "bar", "psi",
            "eV", "keV", "MeV", "GeV", "TeV",
            "calorie", "kilocalorie", "Calorie", "btu", "erg",
            "horsepower", "dyne", "pound_force", "lbf",
            "degree", "arcmin", "arcsec",
            "g", "gravity", "T_bp", "T_fp", "zero_Celsius",
            "R_sun", "M_sun", "L_sun", "R_earth", "M_earth"
        };
    }

    /**
     * Print a summary of constants.
     */
    public static void printSummary() {
        System.out.println("SciPy4J Constants (selection):");
        System.out.println("  pi = " + pi);
        System.out.println("  e = " + e);
        System.out.println("  c = " + c + " m/s");
        System.out.println("  h = " + h + " J s");
        System.out.println("  k_B = " + k_B + " J/K");
        System.out.println("  N_A = " + N_A + " mol^-1");
        System.out.println("  R = " + R + " J/(mol K)");
        System.out.println("  R_inf = " + R_inf + " m^-1");
        System.out.println("  F = " + F + " C/mol");
        System.out.println("  alpha = " + alpha);
        System.out.println("  G = " + G + " m^3 kg^-1 s^-2");
        System.out.println("Total constants: " + names().length);
    }
}
