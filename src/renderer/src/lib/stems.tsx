import type { StemId, ParentStemId } from '../../../shared/types'
import { SUBSTEM_PARENT } from '../../../shared/types'
import { MicIcon, DrumIcon, BassIcon, WaveIcon, PianoIcon, GuitarIcon, MetronomeIcon } from '../components/Icons'

export interface StemMeta {
  id: StemId
  label: string
  color: string
  icon: React.ReactNode
  parent?: ParentStemId
}

const ALL_META: Record<StemId, StemMeta> = {
  click: { id: 'click', label: 'click', color: '#F59E0B', icon: <MetronomeIcon /> },
  vocals: { id: 'vocals', label: 'vocals', color: '#A78BFA', icon: <MicIcon /> },
  lead_vocals: { id: 'lead_vocals', label: 'lead vocal', color: '#A78BFA', icon: <MicIcon />, parent: 'vocals' },
  backing_vocals: { id: 'backing_vocals', label: 'backing vocal', color: '#C084FC', icon: <MicIcon />, parent: 'vocals' },
  drums: { id: 'drums', label: 'drums', color: '#F87171', icon: <DrumIcon /> },
  drums_kick: { id: 'drums_kick', label: 'drums · kick', color: '#F87171', icon: <DrumIcon />, parent: 'drums' },
  drums_snare: { id: 'drums_snare', label: 'drums · snare', color: '#FB923C', icon: <DrumIcon />, parent: 'drums' },
  drums_cymbals: { id: 'drums_cymbals', label: 'drums · cymbals', color: '#FBBF24', icon: <DrumIcon />, parent: 'drums' },
  bass: { id: 'bass', label: 'bass', color: '#60A5FA', icon: <BassIcon /> },
  guitar: { id: 'guitar', label: 'guitar', color: '#F472B6', icon: <GuitarIcon /> },
  acoustic_guitar: { id: 'acoustic_guitar', label: 'violão (acoustic)', color: '#F472B6', icon: <GuitarIcon />, parent: 'guitar' },
  electric_guitar: { id: 'electric_guitar', label: 'guitarra (electric)', color: '#FB7185', icon: <GuitarIcon />, parent: 'guitar' },
  piano: { id: 'piano', label: 'piano', color: '#FBBF24', icon: <PianoIcon /> },
  piano_low: { id: 'piano_low', label: 'piano · graves', color: '#FBBF24', icon: <PianoIcon />, parent: 'piano' },
  piano_high: { id: 'piano_high', label: 'piano · agudos', color: '#FDE047', icon: <PianoIcon />, parent: 'piano' },
  other: { id: 'other', label: 'other', color: '#34D399', icon: <WaveIcon /> },
  other_pads: { id: 'other_pads', label: 'other · pads', color: '#34D399', icon: <WaveIcon />, parent: 'other' },
  other_lead: { id: 'other_lead', label: 'other · synths', color: '#2DD4BF', icon: <WaveIcon />, parent: 'other' }
}

export const SUBSPLIT_LABELS: Partial<Record<StemId, string>> = {
  vocals: 'Split into Lead Vocal + Backing Vocals',
  guitar: 'Split into Violão (Acoustic) + Guitarra (Electric)',
  drums: 'Split into Kick + Snare + Cymbals',
  piano: 'Split into Left Hand (Low) + Right Hand (High)',
  other: 'Split into Pads/Strings + Synths/Lead'
}

// Instruments separated by Demucs from the source audio
export const SEPARABLE_STEMS: StemId[] = ['vocals', 'drums', 'bass', 'guitar', 'piano', 'other']

// Order of lanes rendered in the player (Click at the very top, sub-stems right next to their parent)
export const PLAYER_STEM_ORDER: StemId[] = [
  'click',
  'vocals',
  'lead_vocals',
  'backing_vocals',
  'drums',
  'drums_kick',
  'drums_snare',
  'drums_cymbals',
  'bass',
  'guitar',
  'acoustic_guitar',
  'electric_guitar',
  'piano',
  'piano_low',
  'piano_high',
  'other',
  'other_pads',
  'other_lead'
]

export { ALL_META as STEM_INFO, PLAYER_STEM_ORDER as PREFERRED_ORDER, SUBSTEM_PARENT }

export function buildStemMeta(available: StemId[]): StemMeta[] {
  return PLAYER_STEM_ORDER.filter((id) => available.includes(id)).map((id) => ALL_META[id])
}
