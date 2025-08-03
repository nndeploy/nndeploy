export interface MenuItem {
  key: string;
  text: string;
  icon: string;
  url?: string; 
  items?: MenuItem[];
}
