import Image from "next/image";

// Single source for the "suni c" character mark. The sidebar brand logo and
// the AI panel chat avatar both render through this component so swapping
// the artwork later only means changing the file here.
type SuniAvatarProps = {
  size?: number;
  className?: string;
};

export default function SuniAvatar({ size = 32, className }: SuniAvatarProps) {
  return (
    <Image
      src="/sk-suni-c-5-character.png"
      alt="SUNI"
      width={size}
      height={size}
      unoptimized
      className={className}
      style={{ borderRadius: "50%", objectFit: "cover" }}
    />
  );
}
