import discord
from discord import app_commands
from discord.ext import commands
from typing import Literal
import os
import sys
import tempfile
import aiohttp
import asyncio
import time

# Add the visualizer directory to path
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_VISUALIZER_DIR = os.path.join(os.path.dirname(_SCRIPT_DIR), "better-nothing-music-visualizer")
if _VISUALIZER_DIR not in sys.path:
    sys.path.insert(0, _VISUALIZER_DIR)

from musicViz import GlyphVisualizerAPI

# Request counter file path
_COUNTER_FILE = os.path.join(_SCRIPT_DIR, "glyph_request_count.json")


class ToGlyphComposer(commands.Cog):
    """Commands for generating Nothing Phone glyph visualizer files."""
    
    # Define the command group
    glyphs = app_commands.Group(
        name="glyphs",
        description="Nothing Phone Glyph Composer commands"
    )
    
    def __init__(self, bot: commands.Bot):
        self.bot = bot
        # Initialize the API once
        try:
            self.visualizer_api = GlyphVisualizerAPI()
            self.available_phones = self.visualizer_api.get_available_phones()
        except Exception as e:
            print(f"[ToGlyphComposer] Failed to initialize visualizer API: {e}")
            self.visualizer_api = None
            self.available_phones = []
    
    def _log_request(self, user_id: int, phone: str, duration_ms: float, file_size: int, audio_len: float = 0.0) -> int:
        """Log detailed request info and return the new total count."""
        import json
        from datetime import datetime
        
        entry = {
            "timestamp": datetime.now().isoformat(),
            "user_id": user_id,
            "phone": phone,
            "processing_time_ms": round(duration_ms, 2),
            "file_size_bytes": file_size,
            "audio_length_sec": round(audio_len, 2)
        }
        
        data = {"count": 0, "logs": []}
        try:
            if os.path.isfile(_COUNTER_FILE):
                with open(_COUNTER_FILE, 'r') as f:
                    data = json.load(f)
        except Exception:
            pass
            
        if "logs" not in data:
            data["logs"] = []
            
        data["logs"].append(entry)
        data["count"] = len(data["logs"])
        
        # Keep only last 1000 logs to avoid infinite growth? 
        # User asked for tracking, maybe they want history. Let's keep it for now.
        
        try:
            with open(_COUNTER_FILE, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception:
            pass
            
        return data["count"]
    
    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        """Global check for all commands in this Cog."""
        if hasattr(self.bot, 'server_settings') and interaction.guild_id:
            settings = self.bot.server_settings.get_settings(interaction.guild_id)
            command_name = interaction.command.name
            if not settings.get(f"{command_name}_enabled", True):
                await interaction.response.send_message(
                    "This command is disabled in this server.", 
                    ephemeral=True
                )
                return False
        return True
    
    async def _download_attachment(self, attachment: discord.Attachment, dest_path: str) -> str:
        """Download attachment to local path."""
        async with aiohttp.ClientSession() as session:
            async with session.get(attachment.url) as resp:
                if resp.status == 200:
                    with open(dest_path, 'wb') as f:
                        f.write(await resp.read())
                    return dest_path
                else:
                    raise RuntimeError(f"Failed to download attachment: HTTP {resp.status}")
    
    @glyphs.command(
        name="compose", 
        description="Convert audio to Nothing Phone Glyph Composer format"
    )
    @app_commands.describe(
        audio="Audio file to convert (mp3, m4a, ogg, wav)",
        phone="Phone model for glyph configuration"
    )
    async def compose(
        self,
        interaction: discord.Interaction,
        audio: discord.Attachment,
        phone: Literal["np1", "np1s", "np2", "np2a", "np3a"]
    ):
        """
        Convert an audio file to Nothing Phone Glyph Composer format.
        
        The output OGG file can be imported into Glyph Composer or Glyphify
        for playback on Nothing Phones.
        """
        # Validate API is available
        if self.visualizer_api is None:
            await interaction.response.send_message(
                "Glyph visualizer is not properly configured. Please contact an admin.",
                ephemeral=True
            )
            return
        
        # Validate file type
        valid_extensions = ('.mp3', '.m4a', '.ogg', '.wav', '.flac', '.aac')
        filename_lower = audio.filename.lower()
        if not any(filename_lower.endswith(ext) for ext in valid_extensions):
            await interaction.response.send_message(
                f"Invalid file type. Supported formats: {', '.join(valid_extensions)}",
                ephemeral=True
            )
            return
        
        # Validate file size (max 50MB)
        max_size = 50 * 1024 * 1024
        if audio.size > max_size:
            await interaction.response.send_message(
                f"File too large. Maximum size is 50MB, your file is {audio.size / (1024*1024):.1f}MB",
                ephemeral=True
            )
            return
        
        # Defer response since processing takes time
        await interaction.response.defer(thinking=True)
        
        # Get phone info for response
        try:
            phone_info = self.visualizer_api.get_phone_info(phone)
        except Exception:
            phone_info = {"description": phone, "zone_count": "?"}
        
        temp_dir = None
        try:
            # Create temp directory
            temp_dir = tempfile.mkdtemp(prefix="glyph_cmd_")
            
            # Download the audio file
            input_path = os.path.join(temp_dir, audio.filename)
            await self._download_attachment(audio, input_path)
            
            # Get title from filename (without extension)
            title = os.path.splitext(audio.filename)[0]
            
            # Output path
            output_path = os.path.join(temp_dir, f"{title}_glyph.ogg")
            
            # Measure processing time
            start_time = time.time()
            
            # Run the processing in a thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            result_path = await loop.run_in_executor(
                None,
                lambda: self.visualizer_api.generate_glyph_ogg(
                    audio_path=input_path,
                    phone_model=phone,
                    output_path=output_path,
                    title=title,
                )
            )
            
            processing_time_ms = (time.time() - start_time) * 1000
            
            # Check if output exists
            if not os.path.isfile(result_path):
                await interaction.followup.send(
                    "Failed to generate glyph file. Processing error occurred.",
                    ephemeral=True
                )
                return
            
            # Send the result
            file_size_bytes = os.path.getsize(result_path)
            file_size_kb = file_size_bytes / 1024
            
            # Estimate audio duration from file size (Opus ~112kbps) or just use 0 if unknown
            # Better: get it from visualizer API if possible, but for now 0 is fine or approx.
            # 112kbps = 14KB/s. Duration = Size / 14.
            audio_duration_sec = file_size_kb / 14.0 
            
            # Log detailed request
            request_num = self._log_request(
                user_id=interaction.user.id,
                phone=phone,
                duration_ms=processing_time_ms,
                file_size=file_size_bytes,
                audio_len=audio_duration_sec
            )
            
            embed = discord.Embed(
                title="Glyph Composition Ready",
                description=(
                    f"**Original:** {audio.filename}\n"
                    f"**Phone:** {phone} - {phone_info['description']}\n"
                    f"**Zones:** {phone_info['zone_count']}"
                    f"\n"
                    f"**Output:** {title}_glyph.ogg\n"
                ),
                color=discord.Color.green()
            )
            embed.set_footer(text=f"Output size: {file_size_kb:.1f}KB | Time: {processing_time_ms:.0f}ms")
            
            credits = (
                f"-# Request #{request_num}\n"
                "-# Script by <@1086040015093649499>\n"
                "-# [GitHub](https://github.com/Aleks-Levet/better-nothing-music-visualizer) "
                "| [Discord Thread](https://discord.com/channels/930878214237200394/1434923843239280743)"
            )
            
            await interaction.followup.send(
                content=credits,
                embed=embed,
            )

            # Send another message as ephemeral with the audio (hi sebi if you're checking this out lol)
            await interaction.followup.send(
                file=discord.File(result_path, filename=f"{title}_glyph.ogg"),
                ephemeral=True
            )
            
        except Exception as e:
            error_msg = str(e)
            if len(error_msg) > 200:
                error_msg = error_msg[:200] + "..."
            
            await interaction.followup.send(
                f"Error processing audio:\n```{error_msg}```",
                ephemeral=True
            )
        finally:
            # Clean up temp directory
            if temp_dir and os.path.isdir(temp_dir):
                try:
                    import shutil
                    shutil.rmtree(temp_dir)
                except Exception:
                    pass
    
    @glyphs.command(
        name="list",
        description="List available phone models for Glyph Composer"
    )
    async def list_phones(self, interaction: discord.Interaction):
        """List all available phone models and their zone configurations."""
        if self.visualizer_api is None:
            await interaction.response.send_message(
                "Glyph visualizer is not properly configured.",
                ephemeral=True
            )
            return
        
        embed = discord.Embed(
            title="Available Phone Models",
            description="Use these with `/glyphs compose phone:<model>`",
            color=discord.Color.blue()
        )
        
        for phone_key in self.available_phones:
            try:
                info = self.visualizer_api.get_phone_info(phone_key)
                embed.add_field(
                    name=f"`{phone_key}`",
                    value=f"{info['description']}\n{info['zone_count']} zones",
                    inline=True
                )
            except Exception:
                embed.add_field(
                    name=f"`{phone_key}`",
                    value="Configuration error",
                    inline=True
                )
        
        await interaction.response.send_message(embed=embed)


async def setup(bot: commands.Bot):
    await bot.add_cog(ToGlyphComposer(bot))
