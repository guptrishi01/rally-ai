import { useRef, useState } from 'react'
import { motion } from 'motion/react'

interface DropzoneProps {
  /** Form field name — this renders a real <input type="file">, so the
   *  surrounding form's FormData picks it up with no extra wiring. */
  name: string
  label: string
  hint?: string
  accept?: string
  multiple?: boolean
  /** Reports the current selection's filenames, for callers that show a
   *  preview of what will happen to them (see Overview's part pairing). */
  onNamesChange?: (names: string[]) => void
}

/**
 * A drop target that is a genuine file input underneath.
 *
 * Dropped files are assigned onto the hidden input's own `files` property
 * rather than kept in React state, which is what keeps `new FormData(form)`
 * in Overview working unchanged for both dropping and clicking. Reset by the
 * form's own reset(), same as every other field.
 *
 * Deliberately not `required`: a display:none required control makes Chrome
 * abort the submit with "not focusable" instead of showing anything, so the
 * caller validates the file's presence in its own submit handler.
 */
export default function Dropzone({
  name,
  label,
  hint,
  accept,
  multiple = false,
  onNamesChange,
}: DropzoneProps) {
  const inputRef = useRef<HTMLInputElement>(null)
  const [over, setOver] = useState(false)
  const [names, setNames] = useState<string[]>([])

  function readNames(list: FileList | null) {
    const next = list ? Array.from(list).map((f) => f.name) : []
    setNames(next)
    onNamesChange?.(next)
  }

  function handleDrop(event: React.DragEvent<HTMLDivElement>) {
    event.preventDefault()
    setOver(false)
    const input = inputRef.current
    if (!input || event.dataTransfer.files.length === 0) return
    if (multiple && input.files && input.files.length > 0) {
      // A second drop adds to the selection rather than replacing it -
      // a multi-part match's exports often arrive in more than one go.
      const merged = new DataTransfer()
      for (const file of Array.from(input.files)) merged.items.add(file)
      for (const file of Array.from(event.dataTransfer.files)) merged.items.add(file)
      input.files = merged.files
    } else {
      input.files = event.dataTransfer.files
    }
    readNames(input.files)
  }

  return (
    <div>
      <span className="field-label">{label}</span>
      <motion.div
        onDragOver={(e) => {
          e.preventDefault()
          setOver(true)
        }}
        onDragLeave={() => setOver(false)}
        onDrop={handleDrop}
        onClick={() => inputRef.current?.click()}
        data-cursor="grow"
        animate={{ scale: over ? 1.015 : 1 }}
        transition={{ type: 'spring', stiffness: 320, damping: 24 }}
        className="cursor-pointer rounded-[11px] border border-dashed px-4 py-6 text-center transition-colors"
        style={{
          borderColor: over ? 'var(--color-accent)' : 'var(--color-line-strong)',
          background: over ? 'color-mix(in srgb, var(--color-accent) 6%, transparent)' : 'var(--color-paper)',
        }}
      >
        <input
          ref={inputRef}
          type="file"
          name={name}
          accept={accept}
          multiple={multiple}
          onChange={(e) => readNames(e.target.files)}
          className="hidden"
        />
        {names.length > 0 ? (
          <div className="flex flex-col items-center gap-1">
            {names.map((n) => (
              <span key={n} className="num text-[12px] text-[var(--color-ink)]">
                {n}
              </span>
            ))}
            <span className="eyebrow mt-1">Click to replace</span>
          </div>
        ) : (
          <div className="flex flex-col items-center gap-1.5">
            <span className="text-[13px] text-[var(--color-ink)]">
              Drop {multiple ? 'files' : 'a file'} here, or click to browse
            </span>
            {hint && <span className="eyebrow">{hint}</span>}
          </div>
        )}
      </motion.div>
    </div>
  )
}
