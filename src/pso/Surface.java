/*
 * Copyright 2012 Adrian Toncean
 * 
 * This file is part of OCLEx.
 *
 * OCLEx is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * OCLEx is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with OCLEx. If not, see <http://www.gnu.org/licenses/>.
 */

package pso;

import java.awt.Color;
import org.lwjgl.opengl.GL11;

public class Surface 
{
 public static int handle;
 
 public static void assemble(float[][] val, float minc, float maxc, int rezx, int rezy)
 {
  int i, j;
  float px, py;
  float tmp;

  Color col;

  float spx = -rezx/2;
  float spy = -rezy/2;

  handle = GL11.glGenLists(1);
  GL11.glNewList(handle,GL11.GL_COMPILE);
  
  GL11.glBegin(GL11.GL_QUADS);
  
  px = spx;
  for(i=0;i<rezx-1;i++)
  {
   py = spy;
   for(j=0;j<rezy-1;j++)
   {
    tmp = (float)(1-((val[i][j]-minc) / (maxc-minc)));
    tmp = tmp * 66/100 + 34/100;
    col = new Color(Color.HSBtoRGB(tmp, 0.7f, 0.85f));
    GL11.glColor3f((float)(col.getRed()/255f),(float)(col.getGreen()/255f),(float)(col.getBlue()/255f));

    GL11.glVertex2f(  px,   py);
    
    tmp = (float)(1-((val[i][j+1]-minc) / (maxc-minc)));
    tmp = tmp * 66/100 + 34/100;
    col = new Color(Color.HSBtoRGB(tmp, 0.7f, 0.85f));
    GL11.glColor3f((float)(col.getRed()/255f),(float)(col.getGreen()/255f),(float)(col.getBlue()/255f));
    
    GL11.glVertex2f(  px, py+1);
    
    tmp = (float)(1-((val[i+1][j+1]-minc) / (maxc-minc)));
    tmp = tmp * 66/100 + 34/100;
    col = new Color(Color.HSBtoRGB(tmp, 0.7f, 0.85f));
    GL11.glColor3f((float)(col.getRed()/255f),(float)(col.getGreen()/255f),(float)(col.getBlue()/255f));
    
    GL11.glVertex2f(px+1, py+1);
    
    tmp = (float)(1-((val[i+1][j]-minc) / (maxc-minc)));
    tmp = tmp * 66/100 + 34/100;
    col = new Color(Color.HSBtoRGB(tmp, 0.7f, 0.85f));
    GL11.glColor3f((float)(col.getRed()/255f),(float)(col.getGreen()/255f),(float)(col.getBlue()/255f));
    
    GL11.glVertex2f(px+1,   py);

    py += 1;
   }
   px += 1;
  }

  GL11.glEnd();
  GL11.glEndList(); 
 }
 
 static void call()
 {
  GL11.glCallList(handle);
 }
 
 public static void release()
 {
  GL11.glDeleteLists(handle,1);
 }
}
