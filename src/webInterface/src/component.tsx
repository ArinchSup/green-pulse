import {NavItems} from './variable.ts'
import type { Dispatch, SetStateAction } from 'react'

type OptionProps = {
  activePage: string
  setActivePage: Dispatch<SetStateAction<string>>
}

export const Option = ({ activePage, setActivePage }: OptionProps) => {
  return (
    NavItems.map(item => (
              <div className={`sidebarOP ${activePage === item.id ? 'active' : ''}`} key={item.id} onClick={() => setActivePage(item.id)}>
                {item.label}
              </div>
            ))
    );
};