from tqdm import tqdm
import copy

METAL = ['DDH', 'BEF', 'H79', '1PT', 'AV2', 'P9G', 'SNF', 'CFC', 'HCB', 'BF4',
         'CAS', 'TIL', 'A72', 'IR3', 'HDE', '2J0', 'RHX', 'MYQ', 'PBM', 'SQ1',
         '72B', 'HEG', 'CON', '0UE', 'CPO', 'ALB', 'RKP', 'VER', 'F4S', 'FMI',
         'DRB', '118', 'HEM', '3UQ', 'COB', 'PLL', 'ZPT', 'S31', 'CPT', 'CNC',
         'HEB', 'IUM', 'JSE', 'B1M', 'PCU', 'HIF', 'S32', 'KSB', 'CVC', 'HRU',
         'PCL', '1MK', 'RHD', 'AC9', 'WO3', 'HEO', 'V', 'PT4', 'H58', 'F3S',
         'AM', 'ALF', 'DTZ', '31Q', 'HCO', 'M43', '0JC', 'WO5', 'PT', 'FCE',
         'CU1', 'R7U', 'CU', 'AMW', '0TE', '8WV', 'DVW', 'B9F', 'A71', 'I83',
         'PC4', 'CBY', 'RKL', 'CUL', 'EU3', 'DHE', 'TTO', 'MD9', 'PC3', 'RKM',
         '51O', 'ELJ', 'CL7', 'WO2', 'AF3', '6CQ', 'BVA', 'N7H', 'RH3', 'CUS',
         'JSD', 'PD', 'RE', 'HEV', 'TBY', 'C4R', 'QHL', 'XAX', '2GO', 'MN',
         'DAZ', 'MM1', 'CS', 'MM4', 'HES', 'OS', '73M', 'MM6', 'CAC', 'HBF',
         'SM', 'NCP', 'CA', 'MOS', 'BVQ', 'HG', '7G4', 'HGB', 'IRI', 'ART',
         'ZEM', 'APW', 'PR', 'RUA', 'R6A', 'HNN', 'FES', 'FDD', '35N', 'DW2',
         '5IR', 'GBF', '522', 'OBV', 'H57', 'U1', 'CL1', 'PMR', 'CM2', 'PB',
         'C7P', 'NI', 'HAS', 'EMT', 'A6R', 'GB0', 'ZN9', '3T3', 'MGF', 'VN4',
         'REJ', 'CE', 'IMF', 'YT3', 'COY', 'CSR', 'BE7', 'BCB', '4IR', 'FEC',
         'CAD', 'UVC', 'PHG', '2FH', 'CMH', 'GA', '0TN', 'UFE', 'AUF', 'BTF',
         'TB', 'LOS', 'RIR', '0H2', 'AU3', 'NTE', '6CO', 'CLA', 'BA', 'BAZ',
         'CL0', 'SR', 'AIV', 'MBO', '89R', 'OMO', 'OS1', 'SF3', 'FLL', '4A6',
         'LA', 'IR', 'TPT', 'DW5', '3CG', 'PA0', 'DY', 'HE6', 'EFE', '6B6',
         '9RU', 'ZN7', 'HDD', '5AU', 'CCH', 'RU1', 'CO', 'HEA', 'COH', 'AUC',
         '4KV', 'F43', 'SI4', 'VN3', 'HEC', 'RPS', 'I42', 'ME3', 'HDM', 'R1Z',
         'HME', 'FC6', 'IME', 'J1R', 'GD3', '2PT', '3WB', '6BP', 'RSW', 'POR',
         '9QB', 'RUX', 'ZNH', 'AG1', 'OSV', 'OFE', 'CR', '4MO', 'DWC', 'CAF',
         'PFC', '9D7', '8AR', 'FE', 'MNR', 'OS4', 'B13', 'NA', '3NI', 'RHE',
         'RTA', 'WO4', 'GIX', 'B12', 'CD', 'ZCM', 'CM1', 'RU8', 'ARS', 'BOZ',
         'CUP', '4HE', 'V5A', '9ZQ', 'C5L', '08T', 'SIR', 'YPT', 'YB2', 'MOO',
         'OHX', '7GE', 'AVC', 'MP1', 'LI', 'SF4', 'B22', 'RFB', 'FCI', 'ER3',
         'BF2', 'RUR', 'B30', 'MTQ', 'RU2', '6ZJ', 'GD', 'DEF', '3Q7', 'DW1',
         'MM5', 'YBT', 'YOK', 'BCL', 'ASR', 'I2A', 'SI7', 'Y1', 'CHL', 'RH',
         '6O0', 'SI8', 'N2R', 'PTN', 'REI', 'B1R', 'ITM', 'MOM', 'SI0', 'SVP',
         'RMD', 'MG', 'T0M', 'FE9', '76R', 'RHM', '2MO', 'FS1', 'SI9', '4PU',
         'VVO', '4TI', 'FE2', 'TL', 'UNL', 'CFQ', '2T8', 'RBN', 'MH2', 'VEA',
         'NOB', 'T1A', 'RU', 'MMC', '8CY', 'RKF', 'MTV', 'PL1', '6HE', 'ISW',
         'HE5', 'PMB', 'SB', 'RU7', 'REP', 'RTB', 'ZN6', 'GCR', 'REQ', 'L2D',
         'AL', '9Q8', '1Y8', 'HFM', 'RTC', 'R1C', 'CF', 'MH0', 'CSB', 'L4D',
         'L2M', 'AU', 'AOH', 'AG', 'ZND', 'DRU', 'KYT', 'TTA', 'CX8', 'EU',
         'ATS', '6WO', 'NRU', 'HG2', 'TCN', 'REO', 'QPT', 'SMO', 'N2W', '3ZZ',
         'DEU', 'YOL', 'KYS', 'NXC', 'LU', 'SBO', 'ZN8', 'EMC', 'HKL', 'R4A',
         'MM2', '7BU', '0OD', 'JM1', '6MO', 'YXX', 'JSC', 'HGD', 'SKZ', 'BS3',
         'FEL', 'PNI', 'HB1', 'RHL', 'RML', 'ZN0', 'TH', 'ZN', 'MSS', 'YB',
         'TA0', 'N2N', 'RBU', 'DOZ', 'NCO', 'K', 'RXO', 'YOM', 'FDE', 'LPT',
         '188', 'RAX', 'BJ5', 'CU6', 'RUI', 'IN', 'SXC', 'OEY', 'LSI', 'CX3',
         'PCD', 'RFA', 'ICA', 'B6F', 'MN3', 'MNH', 'RUC', 'M10', 'MOW', '11R',
         'HNI', '7MT', 'W', 'HGI', '1FH', 'WPC', 'MAP', 'R9A', 'ZN5', '68G',
         'RUH', 'RCZ', 'E52', 'HO3', 'MO', '7HE', 'VO4', 'FEM', 'T9T', 'CB5',
         'HO', '6WF', '3CO', 'CLN', '5LN', 'RB', 'TAS', 'CQ4', 'B1Z']

######################################
# CLASS
######################################


class Donor():
    """This class defines a donor with:

    - (flt) distance
    - (int) atom pdb number
    - (str) atom pdb name
    - (str) atom symbol
    - (str) interaction type

    """

    def __init__(self):
        """Constructor of the class."""
        self._distance = 0.00
        self._atom_id = 0
        self._atom_name = ''
        self._element = ''
        self._interaction = ''

    def getDistance(self):
        """Return a copy of distance set."""
        return self._distance

    def setDistance(self, distance):
        """Set distance."""
        self._distance = distance

    def getAtomid(self):
        """Return a copy of atom id set."""
        return self._atom_id

    def setAtomid(self, atom_id):
        """Set atom id."""
        self._atom_id = atom_id

    def getAtomName(self):
        """Return a copy of atom name set."""
        return self._atom_name

    def setAtomName(self, atom_name):
        """Set atom name."""
        self._atom_name = atom_name

    def getElement(self):
        """Return a copy of element set."""
        return self._element

    def setElement(self, element):
        """Set element."""
        self._element = element

    def getInteraction(self):
        """Return a copy of interaction set."""
        return self._interaction

    def setInteraction(self, interaction):
        """Set interaction."""
        self._interaction = interaction


class Ligand():
    """This class defines a ligand with:

    - (int) residue pdb number
    - (str) residue name
    - (str) chain letter
    - (str) endo exo
    - (cls) donor object

    """

    def __init__(self):
        """Constructor of the class."""
        self._residue_id = 0
        self._residue_name = ''
        self._chain_id = ''
        self._endo_exo = ''
        self._donor = Donor()

    def getResidueid(self):
        """Return a copy of residue id set."""
        return self._residue_id

    def setResidueid(self, residue_id):
        """Set residue id."""
        self._residue_id = residue_id

    def getResidueName(self):
        """Return a copy of residue name set."""
        return self._residue_name

    def setResidueName(self, residue_name):
        """Set residue name."""
        self._residue_name = residue_name

    def getChainid(self):
        """Return a copy of chain identifier set."""
        return self._chain_id

    def setChainid(self, chain_id):
        """Set chain identifier."""
        self._chain_id = chain_id

    def getEndoExo(self):
        """Return a copy of endo exo set."""
        return self._endo_exo

    def setEndoExo(self, endo_exo):
        """Set endo exo."""
        self._endo_exo = endo_exo

    def getDonor(self):
        """Return a copy of donor set."""
        return self._donor

    def setDonor(self, donor):
        """Set endo exo."""
        self._donor = donor


class Metal():
    """This class defines a metal with:

    - (str) periodic_symbol
    - (int) coordination number
    - (str) chain letter
    - (str) geometry
    - (str) atom pdb name
    - (int) residue pdb number
    - (str) residue name
    - (str) periodic name
    - (int) atom pdb number
    - (lst) list of ligand object

    """

    def __init__(self):
        """Constructor of the class."""
        self._element = ''
        self._coord = 0
        self._chain_id = ''
        self._geometry = ''
        self._atom_name = ''
        self._residue_id = 0
        self._residue_name = ''
        self._periodic = ''
        self._atom_id = 0
        self._ligands = []

    def getElement(self):
        """Return a copy of element set."""
        return self._element

    def setElement(self, element):
        """Set element."""
        self._element = element

    def getCoord(self):
        """Return a copy of coordination number set."""
        return self._coord

    def setCoord(self, coord):
        """Set coordination number."""
        self._coord = coord

    def getChainid(self):
        """Return a copy of chain identifier set."""
        return self._chain_id

    def setChainid(self, chain_id):
        """Set chain identifier."""
        self._chain_id = chain_id

    def getGeom(self):
        """Return a copy of geometry set."""
        return self._geom

    def setGeom(self, geom):
        """Set geometry."""
        self._geom = geom

    def getAtomName(self):
        """Return a copy of atom name set."""
        return self._atom_name

    def setAtomName(self, atom_name):
        """Set atom name."""
        self._atom_name = atom_name

    def getResidueid(self):
        """Return a copy of residue id set."""
        return self._residue_id

    def setResidueid(self, residue_id):
        """Set residue id."""
        self._residue_id = residue_id

    def getResidueName(self):
        """Return a copy of residue name set."""
        return self._residue_name

    def setResidueName(self, residue_name):
        """Set residue name."""
        self._residue_name = residue_name

    def getPeriodic(self):
        """Return a copy of periodic name set."""
        return self._residue_name

    def setPeriodic(self, periodic):
        """Set residue name."""
        self._periodic = periodic

    def getAtomid(self):
        """Return a copy of atom id set."""
        return self._distance

    def setAtomid(self, atom_id):
        """Set atom id."""
        self._atom_id = atom_id

    def getLigands(self):
        """Return a copy of ligands list set."""
        return self._ligands

    def setLigand(self, ligand):
        """Set ligand list."""
        self._ligands.append(ligand)


class Site_Chain():
    """This class defines a site chain with:

    - (str) molecule name
    - (str) pdb name
    - (str) molecule type
    - (str) letter

    """

    def __init__(self):
        """Constructor of the class."""
        self._molecule_name = ''
        self._pdb_name = ''
        self._molecule_type = ''
        self._letter = ''

    def getMolName(self):
        """Return a copy of molecule name set."""
        return self._molecule_name

    def setMolName(self, molecule_name):
        """Set molecule name."""
        self._molecule_name = molecule_name

    def getPdbName(self):
        """Return a copy of pdb name set."""
        return self._pdb_name

    def setPdbName(self, pdb_name):
        """Set pdb name."""
        self._pdb_name = pdb_name

    def getMolType(self):
        """Return a copy of molecule type set."""
        return self._molecule_type

    def setMolType(self, molecule_type):
        """Set molecule type."""
        self._molecule_type = molecule_type

    def getLetter(self):
        """Return a copy of letter set."""
        return self._letter

    def setLetter(self, letter):
        """Set letter."""
        self._letter = letter


class Site():
    """This class defines a site with:

    - (str) site name
    - (str) pdb code
    - (str) site nuclearity
    - (str) site location
    - (cls) site chain object
    - (cls) metal object

    """

    def __init__(self):
        """Constructor of the class."""
        self._site_name = ''
        self._pdb_code = ''
        self._site_nuc = ''
        self._site_loc = ''
        self._site_chain = Site_Chain()
        self._metal = Metal()

    def getSite(self):
        """Return a copy of site name set."""
        return self._site_name

    def setSite(self, site_name):
        """Set site name."""
        self._site_name = site_name

    def getPDBid(self):
        """Return a copy of pdb code set."""
        return self._pdb_code

    def setPDBid(self, pdb_code):
        """Set pdb code."""
        self._pdb_code = pdb_code

    def getSiteNuc(self):
        """Return a copy of site nuclearity set."""
        return self._site_nuc

    def setSiteNuc(self, site_nuc):
        """Set site nuclearity."""
        self._site_nuc = site_nuc

    def getSiteLoc(self):
        """Return a copy of site location set."""
        return self._site_loc

    def setSiteLoc(self, site_loc):
        """Set site location."""
        self._site_loc = site_loc

    def getSiteChain(self):
        """Return a copy of site chain object set."""
        return self._site_chain

    def setSiteChain(self, site_chain):
        """Set site chain object."""
        self._site_chain = site_chain

    def getMetal(self):
        """Return a copy of metal object set."""
        return self._metal

    def setMetal(self, metal):
        """Set metal object."""
        self._metal = metal
        
def parse_xml(filename, metal_query=None):

    f = open(filename)

    lines = f.readlines()
    i = 0

    flag_metal1 = 0
    flag_metal2 = 0
    flag_metal3 = 0
    flag_metal4 = 0
    flag_metal5 = 0

    site = None
    site_chain = None
    metal = None
    ligand = None
    donor = None

    sites = []

    metals = []

    for i in tqdm(range(len(lines))):

        if "<site>" in lines[i]:
            site = Site()

        if site:
            if "<site_name>" in lines[i]:
                site_name = lines[i].split("<site_name>")[1]
                site_name = site_name.split("</site_name>")[0]
                site.setSite(site_name)
            if "<pdb_code>" in lines[i]:
                pdb_code = lines[i].split("<pdb_code>")[1]
                pdb_code = pdb_code.split("</pdb_code>")[0]
                site.setPDBid(pdb_code)
            if "<site_nuclearity>" in lines[i]:
                site_nuc = lines[i].split("<site_nuclearity>")[1]
                site_nuc = site_nuc.split("</site_nuclearity>")[0]
                site.setSiteNuc(site_nuc)
            if "<site_location>" in lines[i]:
                site_loc = lines[i].split("<site_location>")[1]
                site_loc = site_loc.split("</site_location>")[0]
                site.setSiteLoc(site_loc)

        if "<site_chain>" in lines[i]:
            site_chain = Site_Chain()

        if site_chain:
            if "<molecule_name>" in lines[i]:
                mol_name = lines[i].split("<molecule_name>")[1]
                mol_name = mol_name.split("</molecule_name>")[0]
                site_chain.setMolName(mol_name)
            if "<pdb_name>" in lines[i]:
                pdb_name = lines[i].split("<pdb_name>")[1]
                pdb_name = pdb_name.split("</pdb_name>")[0]
                site_chain.setPdbName(pdb_name)
            if "<molecule_type>" in lines[i]:
                mol_type = lines[i].split("<molecule_type>")[1]
                mol_type = mol_type.split("</molecule_type>")[0]
                site_chain.setMolType(mol_type)
            if "<letter>" in lines[i]:
                letter = lines[i].split("<letter>")[1]
                letter = letter.split("</letter>")[0]
                site_chain.setLetter(letter)

        if "<metal>" in lines[i]:
            metal = Metal()

        if metal:
            if "<periodic_symbol>" in lines[i]:
                element = lines[i].split("<periodic_symbol>")[1]
                element = element.split("</periodic_symbol>")[0]
                metal.setElement(element)
            if "<coordination_number>" in lines[i]:
                coord = lines[i].split("<coordination_number>")[1]
                coord = coord.split("</coordination_number>")[0]
                metal.setCoord(coord)
            if "<chain_letter>" in lines[i] and not flag_metal1:
                chid = lines[i].split("<chain_letter>")[1]
                chid = chid.split("</chain_letter>")[0]
                metal.setChainid(chid)
                flag_metal1 = 1
            if "<geometry>" in lines[i]:
                geom = lines[i].split("<geometry>")[1]
                geom = geom.split("</geometry>")[0]
                metal.setGeom(geom)
            if "<atom_pdb_name>" in lines[i] and not flag_metal2:
                atom_name = lines[i].split("<atom_pdb_name>")[1]
                atom_name = atom_name.split("</atom_pdb_name>")[0]
                metal.setAtomName(atom_name)
                flag_metal2 = 1
            if "<residue_pdb_number>" in lines[i] and not flag_metal3:
                resid = lines[i].split("<residue_pdb_number>")[1]
                resid = resid.split("</residue_pdb_number>")[0]
                metal.setResidueid(resid)
                flag_metal3 = 1
            if "<residue_name>" in lines[i] and not flag_metal4:
                resname = lines[i].split("<residue_name>")[1]
                resname = resname.split("</residue_name>")[0]
                metal.setResidueName(resname)
                flag_metal4 = 1
            if "<periodic_name>" in lines[i]:
                periodic = lines[i].split("<periodic_name>")[1]
                periodic = periodic.split("</periodic_name>")[0]
                metal.setPeriodic(periodic)
            if "<atom_pdb_number>" in lines[i] and not flag_metal5:
                atomid = lines[i].split("<atom_pdb_number>")[1]
                atomid = atomid.split("</atom_pdb_number>")[0]
                metal.setElement(atomid)
                flag_metal5 = 1

        if "<ligand>" in lines[i]:
            ligand = Ligand()

        if ligand:
            if "<residue_pdb_number>" in lines[i]:
                resid = lines[i].split("<residue_pdb_number>")[1]
                resid = resid.split("</residue_pdb_number>")[0]
                ligand.setResidueid(resid)
            if "<residue_name>" in lines[i]:
                resname = lines[i].split("<residue_name>")[1]
                resname = resname.split("</residue_name>")[0]
                ligand.setResidueName(resname)
            if "<chain_letter>" in lines[i]:
                chid = lines[i].split("<chain_letter>")[1]
                chid = chid.split("</chain_letter>")[0]
                ligand.setChainid(chid)
            if "<endo_exo>" in lines[i]:
                endo_exo = lines[i].split("<endo_exo>")[1]
                endo_exo = endo_exo.split("</endo_exo>")[0]
                ligand.setEndoExo(endo_exo)

        if "<donor>" in lines[i]:
            donor = Donor()

        if donor:
            if "<distance>" in lines[i]:
                dist = lines[i].split("<distance>")[1]
                dist = dist.split("</distance>")[0]
                donor.setDistance(dist)
            if "<atom_pdb_number>" in lines[i]:
                atomid = lines[i].split("<atom_pdb_number>")[1]
                atomid = atomid.split("</atom_pdb_number>")[0]
                donor.setAtomid(atomid)
            if "<atom_pdb_name>" in lines[i]:
                atom_name = lines[i].split("<atom_pdb_name>")[1]
                atom_name = atom_name.split("</atom_pdb_name>")[0]
                donor.setAtomName(atom_name)
            if "<atom_symbol>" in lines[i]:
                element = lines[i].split("<atom_symbol>")[1]
                element = element.split("</atom_symbol>")[0]
                donor.setElement(element)
            if "<interaction_type>" in lines[i]:
                interaction = lines[i].split("<interaction_type>")[1]
                interaction = interaction.split("</interaction_type>")[0]
                donor.setInteraction(interaction)

        if "</donor>" in lines[i] and ligand:
            ligand.setDonor(donor)

        if "</ligand>" in lines[i] and metal:
            metal.setLigand(ligand)

        if "</site_chain>" in lines[i] and site:
            site.setSiteChain(site_chain)

        if "</metal>" in lines[i] and site:
            site.setMetal(metal)

        if "</site>" in lines[i]:
            # to keep only metal_query sites
            if metal_query:
                if metal and metal_query in metal.getResidueName():
                    sites.append(copy.deepcopy(site))

            else:
                sites.append(copy.deepcopy(site))
                # if metal:
                #     metals.append(metal.getResidueName())

            flag_metal1 = 0
            flag_metal2 = 0
            flag_metal3 = 0
            flag_metal4 = 0
            flag_metal5 = 0
            site = None
            site_chain = None
            metal = None
            ligand = None
            donor = None

    # metals = set(metals)
    # metals = list(metals)
    # print(metals)

    return sites